import models
import numpy
import ray
import torch

from self_play import MCTS


@ray.remote
class ReplayBuffer:
    """
    Class which run in a dedicated thread to store played games and generate batch.
    """

    def __init__(self, config, game):
        self.config = config
        self.buffer = []
        self.self_play_count = 0
        self.game = game

    def save_game(self, game_history):
        if len(self.buffer) > self.config.window_size:
            self.buffer.pop(0)
        self.buffer.append(game_history)
        self.self_play_count += 1

    def get_self_play_count(self):
        return self.self_play_count

    def get_batch(self, target_network_weights=None):
        observation_batch, action_batch, reward_batch, value_batch, policy_batch = (
            [],
            [],
            [],
            [],
            [],
        )
        for _ in range(self.config.batch_size):
            game_history = self.sample_game(self.buffer)
            game_pos = self.sample_position(game_history)

            values, rewards, policies, actions = self.make_target(
                game_history, game_pos, target_network_weights
            )

            observation_batch.append(game_history.observation_history[game_pos])
            action_batch.append(actions)
            value_batch.append(values)
            reward_batch.append(rewards)
            policy_batch.append(policies)

        # observation_batch: batch, channels, height, width
        # action_batch: batch, num_unroll_steps+1
        # value_batch: batch, num_unroll_steps+1
        # reward_batch: batch, num_unroll_steps+1
        # policy_batch: batch, num_unroll_steps+1, len(action_space)
        return observation_batch, action_batch, value_batch, reward_batch, policy_batch

    def sample_game(self, buffer):
        """
        Sample game from buffer either uniformly or according to some priority.
        """
        # TODO: sample with probability link to the highest difference between real and
        # predicted value (See paper appendix Training)
        return numpy.random.choice(self.buffer)

    def sample_position(self, game_history):
        """
        Sample position from game either uniformly or according to some priority.
        """
        # TODO: sample according to some priority
        return numpy.random.choice(range(len(game_history.reward_history)))

    def make_target(self, game_history, state_index, target_network_weights=None):
        """
        The value target is the discounted root value of the search tree td_steps into the
        future, plus the discounted sum of all rewards until then.
        """
        target_values, target_rewards, target_policies, actions = [], [], [], []

        if target_network_weights:
            target_network = models.MuZeroNetwork(self.config)
            target_network.set_weights(target_network_weights)
            target_network = target_network.to("cpu")
            target_network.eval()
        else:
            target_network = None

        for current_index in range(
            state_index, state_index + self.config.num_unroll_steps + 1
        ):
            bootstrap_index = current_index + self.config.td_steps
            if bootstrap_index < len(game_history.root_values):
                if target_network is None:
                    value = game_history.root_values[bootstrap_index]
                else:
                    obs = game_history.observation_history[bootstrap_index]
                    obs = (
                        torch.tensor(obs)
                            .float()
                            .unsqueeze(0)
                            .to(next(target_network.parameters()).device)
                    )
                    value, _, _, _ = target_network.initial_inference(obs)
                    value = MCTS.support_to_scalar(value, self.config.support_size).cpu().item()

                value = value * self.config.discount ** self.config.td_steps

            else:
                value = 0

            for i, reward in enumerate(
                game_history.reward_history[current_index:bootstrap_index]
            ):
                value += (
                    reward
                    if game_history.to_play_history[current_index]
                    == game_history.to_play_history[current_index + i]
                    else -reward
                ) * self.config.discount ** i

            if current_index < len(game_history.root_values):
                target_values.append(value)
                target_rewards.append(game_history.reward_history[current_index])

                if target_network is not None and numpy.random.random() <= 0.8:
                    with torch.no_grad():
                        obs = game_history.observation_history[current_index]
                        root = MCTS(self.config).run(target_network, obs, self.game.legal_actions(), self.game.to_play(), False)
                        game_history.store_search_statistics(root, self.config.action_space, current_index)

                target_policies.append(game_history.child_visits[current_index])
                actions.append(game_history.action_history[current_index])
            elif current_index == len(game_history.root_values):
                target_values.append(value)
                target_rewards.append(game_history.reward_history[current_index])
                target_policies.append(
                    [
                        1 / len(game_history.child_visits[0])
                        for _ in range(len(game_history.child_visits[0]))
                    ]
                )
                actions.append(game_history.action_history[current_index])
            else:
                # States past the end of games are treated as absorbing states
                target_values.append(0)
                target_rewards.append(0)
                # Uniform policy to give the tensor a valid dimension
                target_policies.append(
                    [
                        1 / len(game_history.child_visits[0])
                        for _ in range(len(game_history.child_visits[0]))
                    ]
                )
                actions.append(numpy.random.choice(game_history.action_history))

        return target_values, target_rewards, target_policies, actions
