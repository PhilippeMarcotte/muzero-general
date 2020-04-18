import ray
import torch
import os


@ray.remote
class SharedStorage:
    """
    Class which run in a dedicated thread to store the network weights and some information.
    """

    def __init__(self, weights, game_name, config):
        self.config = config
        self.game_name = game_name
        self.target_network_weights = weights
        self.network_weights = weights
        self.infos = {
            "total_reward": 0,
            "average_reward": 0,
            "player_0_reward": 0,
            "player_1_reward": 0,
            "episode_length": 0,
            "training_step": 0,
            "test_games": 0,
            "samples_count": 0,
            "reanalyzed_count": 0,
            "remcts_count": 0,
            "lr": 0,
            "total_loss": 0,
            "value_loss": 0,
            "reward_loss": 0,
            "policy_loss": 0,
        }

    def get_target_network_weights(self):
        return self.target_network_weights

    def set_target_network_weights(self, weights, path=None):
        self.target_network_weights = weights
        if not path:
            path = os.path.join(self.config.results_path, "model.weights")

        torch.save(self.target_network_weights, path)

    def get_network_weights(self):
        return self.network_weights

    def set_network_weights(self, weights):
        self.network_weights = weights

    def get_infos(self):
        return self.infos

    def set_infos(self, key, value):
        self.infos[key] = value

    def update_infos(self, key, value):
        self.infos[key] += value
