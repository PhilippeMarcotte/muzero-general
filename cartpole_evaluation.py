import json
import os
import pathlib
import shutil
import string
import time

import fire
import ray
import wandb
from ray.experimental.queue import Queue

from muzero import MuZero
from utils.config import load_toml
import re


@ray.remote
class SharedResults:
    def __init__(self, num_episodes):
        self.results = {}
        self.num_episodes = num_episodes

    def get_result(self):
        return self.results

    def set_result(self, env_config_name, value):
        if env_config_name not in self.results:
            self.results[env_config_name] = []
        self.results[env_config_name].append(value)


def find_env_config(files, regex):
    for file in files:
        match = re.search(regex, file.name)
        if match is not None:
            return file
    return None


@ray.remote
class ModelEvaluator:
    WEIGHTS_DIR_PATH = "./weights"
    CONFIGS_DIR_PATH = "./games"

    def __init__(self, job_queue, results, num_episodes):
        self.job_queue = job_queue
        self.results = results
        self.num_episodes = num_episodes

    def evaluate(self):
        print(self.job_queue.empty() > 0)
        while self.job_queue.empty() > 0:
            env_config_name, weight_file_path, env_config_file, seed = self.job_queue.get()
            print(weight_file_path)

            muzero = MuZero(env_config_name, seed=seed)
            muzero.load_model(weight_file_path)
            total_reward = muzero.test(render=False, opponent="self", muzero_player=None, ray_init=False)
            print(f"Total reward : {total_reward}")
            ray.get(self.results.set_result.remote(env_config_name, total_reward))
            print(f"{env_config_name} done.")
        return True


def evaluation(evaluation_config_path="./configs/fast_reanalyze_evaluation.toml"):
    t1 = time.time()
    ray.init()
    config = load_toml(evaluation_config_path)
    api = wandb.Api()
    if len(config.run_ids) > 0:
        runs = [api.run(path=f"{config.entity}/{config.project_name}/{id}") for id in config.run_ids]
    else:
        runs = api.runs(path=f"{config.entity}/{config.project_name}", filters=config.filters)
    results = SharedResults.remote(num_episodes=config.num_episodes)

    job_queue = Queue()
    # Fill the queue with models to evaluate
    for run in runs:
        files = run.files()
        print(files)
        env_config_file = find_env_config(files.objects, r"(:?^|\s)\w*(?=.py)")
        try:
            weights_file_result = run.files("model.weights")
            if env_config_file is None:
                continue
            env_config_name = os.path.splitext(env_config_file.name)[0]
            # if os.path.exists(os.path.join(ModelEvaluator.CONFIGS_DIR_PATH, env_config_file.name)) is False:
            env_config_file.download(True, root=ModelEvaluator.CONFIGS_DIR_PATH)
            weight_file_path = os.path.join(ModelEvaluator.WEIGHTS_DIR_PATH, env_config_name, f"{run.id}.weights")
            if os.path.exists(weight_file_path) is False:
                pathlib.Path(os.path.dirname(weight_file_path)).mkdir(parents=True, exist_ok=True)
                weights_file = weights_file_result[0].download(replace=True,
                                                               root=ModelEvaluator.WEIGHTS_DIR_PATH)
                shutil.move(weights_file.name, weight_file_path)
                weight_file_path = weights_file.name
                del weights_file

            for seed in range(config.num_episodes):
                job_queue.put((env_config_name, weight_file_path, env_config_file, seed))
        except:
            print(f"{run.name} failure")

    # Start the model evaluator worker
    evaluators = []
    for _ in range(config.num_workers):
        model_evaluator = ModelEvaluator.remote(job_queue, results, config.num_episodes)
        evaluators.append(model_evaluator.evaluate.remote())
    # Wait for all the workers to be done
    ray.get(evaluators)
    # Save the results
    ids_string = '_'.join(config.run_ids[-10])
    filter_string = '_'.join([f"{key}-{value}" for key, value in config.filters.items()])
    with open(f'evaluation_results/test_results_{ids_string}_{filter_string}.json', 'w') as outfile:
        json.dump(ray.get(results.get_result.remote()), outfile)
    print(f"Time taken : {time.time() - t1}")


if __name__ == "__main__":
    fire.Fire(evaluation)
