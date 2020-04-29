import wandb
from ray.experimental.queue import Queue

import reanalyze
from utils.logging import Logger
import copy
import importlib
import os
import time

import fire
import numpy
import ray
import torch

import models
import replay_buffer
import self_play
import shared_storage
import trainer
import fast_reanalyze
from utils.config import load_toml
from utils.logging import WandbLogger, TensorboardLogger


class MuZero:
    """
    Main class to manage MuZero.

    Args:
        game_name (str): Name of the game module, it should match the name of a .py file
        in the "./games" directory.

    Example:
        >>> muzero = MuZero("cartpole")
        >>> muzero.train()
        >>> muzero.test(render=True, opponent="self", muzero_player=None)
    """

    def __init__(self, game_name, seed=None):
        self.game_name = game_name

        # Load the game and the config from the module with the game name
        try:
            game_module = importlib.import_module("games." + self.game_name)
            self.config = game_module.MuZeroConfig()
            if seed is not None:
                self.config.seed = seed
            self.Game = game_module.Game
        except Exception as err:
            print(
                '{} is not a supported game name, try "cartpole" or refer to the documentation for adding a new game.'.format(
                    self.game_name
                )
            )
            raise err

        # Fix random generator seed for reproductibility
        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        # Weights used to initialize components
        self.muzero_weights = models.MuZeroNetwork(self.config).get_weights()

    def train(self, writer: Logger):
        ray.init()
        os.makedirs(self.config.results_path, exist_ok=True)

        # Initialize workers
        training_worker = trainer.Trainer.options(
            num_gpus=1 if "cuda" in self.config.training_device else 0
        ).remote(copy.deepcopy(self.muzero_weights), self.config)
        shared_storage_worker = shared_storage.SharedStorage.remote(
            copy.deepcopy(self.muzero_weights), self.game_name, self.config,
        )
        replay_buffer_worker = replay_buffer.ReplayBuffer.remote(self.config, shared_storage_worker)
        self_play_workers = [
            self_play.SelfPlay.remote(
                copy.deepcopy(self.muzero_weights),
                self.Game(self.config.seed + seed),
                self.config,
            )
            for seed in range(self.config.num_actors)
        ]
        test_worker = self_play.SelfPlay.remote(
            copy.deepcopy(self.muzero_weights),
            self.Game(self.config.seed + self.config.num_actors),
            self.config,
        )
        queue = None
        if self.config.policy_update_rate > 0:
            if self.config.reanalyze_mode == "fast":
                reanalyze_worker = fast_reanalyze.ReanalyzeWorker.remote(
                    copy.deepcopy(self.muzero_weights),
                    shared_storage_worker,
                    replay_buffer_worker,
                    self.config
                )
                reanalyze_worker.update_policies.remote()
            else:
                queue = Queue()
                for i in range(self.config.num_reanalyze_cpus):
                    reanalyze_worker = reanalyze.ReanalyzeQueueWorker.remote(
                        copy.deepcopy(self.muzero_weights),
                        shared_storage_worker,
                        replay_buffer_worker,
                        self.config,
                        queue
                    )
                    reanalyze_worker.fill_batch_queue.remote()
        # Launch workers
        [
            self_play_worker.continuous_self_play.remote(
                shared_storage_worker, replay_buffer_worker
            )
            for self_play_worker in self_play_workers
        ]
        test_worker.continuous_self_play.remote(shared_storage_worker, None, True)
        training_worker.continuous_update_weights.remote(
            replay_buffer_worker, shared_storage_worker, queue
        )

        # Save hyperparameters to TensorBoard
        hp_table = [
            "| {} | {} |".format(key, value)
            for key, value in self.config.__dict__.items()
        ]
        writer.add_text(
            "Hyperparameters",
            "| Parameter | Value |\n|-------|-------|\n" + "\n".join(hp_table),
        )
        # Loop for monitoring in real time the workers
        counter = 0
        infos = ray.get(shared_storage_worker.get_infos.remote())
        try:
            while infos["training_step"] < self.config.training_steps:
                # Get and save real time performance
                infos = ray.get(shared_storage_worker.get_infos.remote())
                writer.add_scalar(
                    "1.Total reward/1.Total reward", infos["total_reward"], counter,
                )
                writer.add_scalar(
                    "1.Total reward/2.Episode length", infos["episode_length"], counter,
                )
                writer.add_scalar(
                    "1.Total reward/3.Player 0 MuZero reward",
                    infos["player_0_reward"],
                    counter,
                )

                writer.add_scalar(
                    "1.Total reward/4.Player 1 Random reward",
                    infos["player_1_reward"],
                    counter,
                )
                writer.add_scalar(
                    "1.Total reward/5.Average reward", infos["average_reward"], counter,
                )
                writer.add_scalar(
                    "2.Workers/1.Self played games",
                    ray.get(replay_buffer_worker.get_self_play_count.remote()),
                    counter,
                )
                writer.add_scalar(
                    "2.Workers/2.Training steps", infos["training_step"], counter
                )
                writer.add_scalar(
                    "2.Workers/3.Self played games per training step ratio",
                    ray.get(replay_buffer_worker.get_self_play_count.remote())
                    / max(1, infos["training_step"]),
                    counter,
                )
                writer.add_scalar("2.Workers/4.Learning rate", infos["lr"], counter)
                writer.add_scalar(
                    "2.Workers/5.Self played test games",
                    infos["test_games"],
                    counter,
                )
                writer.add_scalar(
                    "2.Workers/6.Samples count per training step ratio",
                    infos["samples_count"]
                    / max(1, infos["training_step"]),
                    counter,
                )
                writer.add_scalar(
                    "2.Workers/7.Samples count",
                    infos["samples_count"],
                    counter,
                )
                writer.add_scalar(
                    "2.Workers/8.Reanalyzed count",
                    infos["reanalyzed_count"],
                    counter,
                )
                writer.add_scalar(
                    "2.Workers/9.Reanalyzed count per samples count",
                    infos["reanalyzed_count"] / max(1, infos["samples_count"]),
                    counter,
                )
                writer.add_scalar(
                    "2.Workers/10.ReMCTS count",
                    infos["remcts_count"],
                    counter,
                )
                writer.add_scalar(
                    "2.Workers/11.ReMCTS count per samples count",
                    infos["remcts_count"] / max(1, infos["samples_count"]),
                    counter,
                )
                writer.add_scalar(
                    "3.Loss/1.Total weighted loss", infos["total_loss"], counter
                )
                writer.add_scalar("3.Loss/Value loss", infos["value_loss"], counter)
                writer.add_scalar("3.Loss/Reward loss", infos["reward_loss"], counter)
                writer.add_scalar("3.Loss/Policy loss", infos["policy_loss"], counter)
                print(
                    "Last test reward: {0:.2f}. Training step: {1}/{2}. Played games: {3}. Loss: {4:.2f}".format(
                        infos["total_reward"],
                        infos["training_step"],
                        self.config.training_steps,
                        ray.get(replay_buffer_worker.get_self_play_count.remote()),
                        infos["total_loss"],
                    ),
                    end="\r",
                )
                counter += 1
                time.sleep(0.5)
        except KeyboardInterrupt as err:
            # Comment the line below to be able to stop the training but keep running
            # raise err
            pass
        self.muzero_weights = ray.get(shared_storage_worker.get_target_network_weights.remote())
        # End running actors
        ray.shutdown()

    def test(self, render, opponent, muzero_player, ray_init=True):
        """
        Test the model in a dedicated thread.

        Args:
            render: Boolean to display or not the environment.

            opponent: "self" for self-play, "human" for playing against MuZero and "random"
            for a random agent.

            muzero_player: Integer with the player number of MuZero in case of multiplayer
            games, None let MuZero play all players turn by turn.
        """
        print("\nTesting...")
        if ray_init:
            ray.init()
        self_play_workers = self_play.SelfPlay.remote(
            copy.deepcopy(self.muzero_weights),
            self.Game(self.config.seed + self.config.num_actors),
            self.config,
        )
        history = ray.get(
            self_play_workers.play_game.remote(0, 0, render, opponent, muzero_player)
        )
        if ray_init:
            ray.shutdown()
        return sum(history.reward_history)

    def load_model(self, path=None):
        if not path:
            path = os.path.join(self.config.results_path, "model.weights")
        try:
            self.muzero_weights = torch.load(path)
            print("\nUsing weights from {}".format(path))
        except FileNotFoundError:
            print("\nThere is no model saved in {}.".format(path))


def main(game_name="reanalyze_cartpole", action="Train", seed=None, tags=[], logger="wandb",
         config_path="./configs/config.toml", group=None):
    """
    Hello

    @param game_name: File name of any games in the game folder
    @param action: ["Train", "Load pretrained model", "Render some self play games", "Play against MuZero"]
    @param logger: wandb or tensorboard
    """
    config = load_toml(config_path)
    print("\nWelcome to MuZero! Here's a list of games:")
    # Let user pick a game
    games = [
        filename[:-3]
        for filename in sorted(os.listdir("./games"))
        if filename.endswith(".py") and filename != "abstract_game.py"
    ]
    for i in range(len(games)):
        print("{}. {}".format(i, games[i]))
    if game_name not in games:
        valid_inputs = [str(i) for i in range(len(games))]
        choice = input("Enter a number to choose the game: ")
        while choice not in valid_inputs:
            choice = input("Invalid input, enter a number listed above: ")
        choice = int(choice)
        game_name = games[choice]

    # Initialize MuZero
    muzero = MuZero(game_name, seed)
    if logger == "wandb":
        if group is not None:
            config.wandb.group = group
        tags.append(f"seed={seed}")
        logger = WandbLogger(config, muzero.config, tags)
        logger.writer.save(f"games/{game_name}.py")
        logger.writer.save("configs/config.toml")
    else:
        logger = TensorboardLogger(config, muzero.config)
    while True:
        # Configure running options
        options = [
            "Train",
            "Load pretrained model",
            "Render some self play games",
            "Play against MuZero",
            "Exit",
        ]
        print()
        for i in range(len(options)):
            print("{}. {}".format(i, options[i]))

        if action not in options:
            choice = input("Enter a number to choose an action: ")
            valid_inputs = [str(i) for i in range(len(options))]
            while choice not in valid_inputs:
                choice = input("Invalid input, enter a number listed above: ")
            choice = int(choice)
        else:
            choice = options.index(action)

        if choice == 0:
            muzero.train(logger)
        elif choice == 1:
            path = input("Enter a path to the model.weights: ")
            while not os.path.isfile(path):
                path = input("Invalid path. Try again: ")
            muzero.load_model(path)
        elif choice == 2:
            muzero.test(render=True, opponent="self", muzero_player=None)
        elif choice == 3:
            muzero.test(render=True, opponent="human", muzero_player=0)
        else:
            break
        print("\nDone")
        if action is not None:
            break

    if isinstance(logger, WandbLogger):
        logger.writer.save(os.path.join(muzero.config.results_path, "model.weights"))
    ## Successive training, create a new config file for each experiment
    # experiments = ["cartpole", "tictactoe"]
    # for experiment in experiments:
    #     print("\nStarting experiment {}".format(experiment))
    #     try:
    #         muzero = MuZero(experiment)
    #         muzero.train()
    #     except:
    #         print("Skipping {}, an error has occurred.".format(experiment))


if __name__ == "__main__":
    fire.Fire(main)
