import random
import time

import ray
import torch

import models
from self_play import MCTS


@ray.remote(num_cpus=1)
class ReanalyzeWorker:
    def __init__(self, initial_weights, shared_storage, replay_buffer, config):
        self.shared_storage = shared_storage
        self.replay_buffer = replay_buffer
        self.config = config
        # Initialize the network
        self.model = models.MuZeroNetwork(self.config)
        self.model.set_weights(initial_weights)
        self.model.to(torch.device("cpu"))
        self.model.eval()

    def update_policies(self):
        while True:
            for game_id in range(ray.get(self.replay_buffer.get_buffer_size.remote())):
                game_history = ray.get(self.replay_buffer.get_game_history.remote(game_id))
                for pos in range(len(game_history.observation_history) - 1):
                    if random.random() < self.config.policy_update_rate:
                        self.model.set_weights(ray.get(self.shared_storage.get_network_weights.remote()))
                        self.model.to(torch.device("cpu"))
                        self.model.eval()
                        with torch.no_grad():
                            stacked_obs = torch.tensor(
                                game_history.get_stacked_observations(
                                    pos, self.config.stacked_observations
                                )
                            ).float()

                            root, _, _ = MCTS(self.config).run(self.model, stacked_obs, game_history.legal_actions[pos],
                                                               game_history.to_play_history[pos], False)
                            game_history.store_search_statistics(root, self.config.action_space, pos)
