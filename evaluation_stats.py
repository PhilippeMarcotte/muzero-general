import json
import numpy as np

import fire


def evaluation_stats(results_path=r'./evaluation_results/final/test_results__group-basic_cartpole.json'):
    with open(results_path, mode="r") as f:
        results = json.load(f)
        mean_scores = {}
    for key, value in results.items():
        # @plelievre : is the right way to compute the mean?
        mean_scores[key] = np.mean(results[key])
    print(mean_scores)


if __name__ == "__main__":
    fire.Fire(evaluation_stats)
