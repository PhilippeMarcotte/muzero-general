import random
import time

import ray
import torch

import models
from self_play import MCTS
import copy

@ray.remote(num_cpus=1)
class ReanalyzeWorker:
    def __init__(self, initial_weights, shared_storage, replay_buffer, config):
        self.shared_storage = shared_storage
        self.replay_buffer = replay_buffer
        self.config = config
        # Initialize the network
        self.latest_network = models.MuZeroNetwork(self.config)
        self.latest_network.set_weights(initial_weights)
        self.latest_network.to(torch.device("cpu"))
        self.latest_network.eval()

        self.target_network = models.MuZeroNetwork(self.config)
        self.target_network.set_weights(initial_weights)
        self.target_network.to(torch.device("cpu"))
        self.target_network.eval()

    def update_policies(self):
        while True:
            keys = ray.get(self.replay_buffer.get_buffer_keys.remote())
            for game_id in keys:
                remcts_count = 0
                self.latest_network.set_weights(ray.get(self.shared_storage.get_network_weights.remote()))
                self.target_network.set_weights(ray.get(self.shared_storage.get_target_network_weights.remote()))

                game_history = copy.deepcopy(ray.get(self.replay_buffer.get_game_history.remote(game_id)))

                for pos in range(len(game_history.observation_history)):
                    bootstrap_index = pos + self.config.td_steps
                    if bootstrap_index < len(game_history.root_values):
                        if self.config.use_last_model_value:
                            # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
                            observation = torch.tensor(
                                game_history.get_stacked_observations(
                                    bootstrap_index, self.config.stacked_observations
                                )
                            ).float()
                            value = models.support_to_scalar(
                                self.target_network.initial_inference(observation)[0],
                                self.config.support_size,
                            ).item()
                            game_history.root_values[bootstrap_index] = value

                    if random.random() < self.config.policy_update_rate and pos < len(game_history.root_values):
                        with torch.no_grad():
                            stacked_obs = torch.tensor(
                                game_history.get_stacked_observations(
                                    pos, self.config.stacked_observations
                                )
                            ).float()

                            root, _, _ = MCTS(self.config).run(self.latest_network, stacked_obs, game_history.legal_actions[pos],
                                                               game_history.to_play_history[pos], False)
                            game_history.store_search_statistics(root, self.config.action_space, pos)
                        remcts_count += 1

                self.shared_storage.update_infos.remote("remcts_count", remcts_count)
                self.shared_storage.update_infos.remote("reanalyzed_count", len(game_history.priorities))
                self.replay_buffer.update_game.remote(game_history, game_id)
