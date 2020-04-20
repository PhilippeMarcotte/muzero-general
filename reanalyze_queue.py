import random
import time

import ray
import torch
from ray.experimental.queue import Queue
import numpy as np
import models
from self_play import MCTS
import copy


@ray.remote(num_cpus=1)
class ReanalyzeQueueWorker:
    def __init__(self, initial_weights, shared_storage, replay_buffer, config, queue):
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

        self.queue = queue

    def prepare_batch(self):
        (
            index_batch,
            observation_batch,
            action_batch,
            reward_batch,
            value_batch,
            policy_batch,
            weight_batch,
            gradient_scale_batch,
        ) = ([], [], [], [], [], [], [], [])

        self.latest_network.set_weights(ray.get(self.shared_storage.get_network_weights.remote()))
        self.target_network.set_weights(ray.get(self.shared_storage.get_target_network_weights.remote()))

        for _ in range(self.config.batch_size):
            total_samples, game_id, game_history, game_prob, game_pos, pos_prob = ray.get(
                self.replay_buffer.sample_game_position.remote())

            values, rewards, policies, actions = self.make_target(
                game_history, game_pos
            )

            index_batch.append([game_id, game_pos])
            observation_batch.append(
                game_history.get_stacked_observations(
                    game_pos, self.config.stacked_observations
                )
            )
            action_batch.append(actions)
            value_batch.append(values)
            reward_batch.append(rewards)
            policy_batch.append(policies)
            weight_batch.append(
                (total_samples * game_prob * pos_prob) ** (-self.config.PER_beta)
            )
            gradient_scale_batch.append(
                [
                    min(
                        self.config.num_unroll_steps,
                        len(game_history.action_history) - game_pos,
                    )
                ]
                * len(actions)
            )

        weight_batch = np.array(weight_batch) / max(weight_batch)

        # observation_batch: batch, channels, height, width
        # action_batch: batch, num_unroll_steps+1
        # value_batch: batch, num_unroll_steps+1
        # reward_batch: batch, num_unroll_steps+1
        # policy_batch: batch, num_unroll_steps+1, len(action_space)
        # weight_batch: batch
        # gradient_scale_batch: batch, num_unroll_steps+1
        return (
            index_batch,
            (
                observation_batch,
                action_batch,
                value_batch,
                reward_batch,
                policy_batch,
                weight_batch,
                gradient_scale_batch,
            ),
        )

    def make_target(self, game_history, state_index):
        """
        Generate targets for every unroll steps.
        """
        target_values, target_rewards, target_policies, actions = [], [], [], []
        for current_index in range(
                state_index, state_index + self.config.num_unroll_steps + 1
        ):
            # The value target is the discounted root value of the search tree td_steps into the
            # future, plus the discounted sum of all rewards until then.
            bootstrap_index = current_index + self.config.td_steps
            if bootstrap_index < len(game_history.root_values):
                if self.config.use_last_model_value:
                    # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
                    observation = torch.tensor(
                        game_history.get_stacked_observations(
                            bootstrap_index, self.config.stacked_observations
                        )
                    ).float()
                    last_step_value = models.support_to_scalar(
                        self.target_network.initial_inference(observation)[0],
                        self.config.support_size,
                    ).item()
                else:
                    last_step_value = game_history.root_values[bootstrap_index]

                value = last_step_value * self.config.discount ** self.config.td_steps
            else:
                value = 0

            for i, reward in enumerate(
                    game_history.reward_history[current_index + 1: bootstrap_index + 1]
            ):
                value += (
                             reward
                             if game_history.to_play_history[current_index]
                                == game_history.to_play_history[current_index + 1 + i]
                             else -reward
                         ) * self.config.discount ** i

            if current_index < len(game_history.root_values):
                if random.random() < self.config.policy_update_rate and current_index < len(game_history.root_values):
                    with torch.no_grad():
                        stacked_obs = torch.tensor(
                            game_history.get_stacked_observations(
                                current_index, self.config.stacked_observations
                            )
                        ).float()

                        root, _, _ = MCTS(self.config).run(self.latest_network, stacked_obs,
                                                           game_history.legal_actions[current_index],
                                                           game_history.to_play_history[current_index], False)
                        game_history.store_search_statistics(root, self.config.action_space, current_index)
                target_values.append(value)
                target_rewards.append(game_history.reward_history[current_index])
                target_policies.append(game_history.child_visits[current_index])
                actions.append(game_history.action_history[current_index])
            elif current_index == len(game_history.root_values):
                target_values.append(0)
                target_rewards.append(game_history.reward_history[current_index])
                # Uniform policy
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
                # Uniform policy
                target_policies.append(
                    [
                        1 / len(game_history.child_visits[0])
                        for _ in range(len(game_history.child_visits[0]))
                    ]
                )
                actions.append(np.random.choice(game_history.action_history))

        return target_values, target_rewards, target_policies, actions

    def fill_batch_queue(self):
        while ray.get(self.replay_buffer.get_self_play_count.remote()) < 1:
            time.sleep(0.1)
        while True:
            # print("before put")
            t1 = time.time()
            batch = self.prepare_batch()
            # print(f"took {time.time() - t1}")
            self.queue.put(batch)
            # print("after put")

    # def get_batch(self):
    #     print(f"getting batch of len {len(self.queue)}")
    #     batch = self.queue.
    #     print("got batch")
    #
    #     return batch
