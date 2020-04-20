import json
import os
import shutil

import fire
import wandb
from muzero import MuZero
from utils.config import load_toml
import re


def find_env_config(files, regex):
    for file in files:
        match = re.search(regex, file.name)
        if match is not None:
            return file
    return None


def evaluation(evaluation_config_path="./configs/evaluation.toml"):
    config = load_toml(evaluation_config_path)
    api = wandb.Api()
    runs = api.runs(path=f"{config.entity}/{config.project_name}", filters=config.filters)
    results = {}
    for run in runs:
        files = run.files()
        print(files)
        env_config_file = find_env_config(files.objects, r"(:?^|\s)\w*(?=.py)")
        try:
            weights_file_result = run.files("model.weights")
            if len(weights_file_result) > 0 and env_config_file is not None:
                weights_file = weights_file_result[0].download(True)
                config_file = env_config_file.download(True)
                shutil.copy(config_file.name, "./games")
                os.remove(config_file.name)
                env_config_name = os.path.splitext(env_config_file.name)[0]
                muzero = MuZero(env_config_name, seed=0)
                muzero.load_model(weights_file.name)
                del weights_file
                total_reward_for_all_episodes = []
                for i in range(config.num_episodes):
                    muzero.config.seed = i
                    total_reward = muzero.test(render=False, opponent="self", muzero_player=None)
                    total_reward_for_all_episodes.append(total_reward)
                print(f"{env_config_name} done.")
                if env_config_name in results:
                    results[env_config_name].append(total_reward_for_all_episodes)
                else:
                    results[env_config_name] = [total_reward_for_all_episodes]
        except:
            print(f"{run.name} failure")
    with open(f'evaluation_results/test_results_{config.filters.toJSON()}.json', 'w') as outfile:
        json.dump(results, outfile)


if __name__ == "__main__":
    fire.Fire(evaluation)
