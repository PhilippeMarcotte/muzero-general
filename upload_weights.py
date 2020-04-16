import wandb
import os
from utils.config import load_toml
import yaml

root = "./wandb"
api = wandb.Api()
for directory in os.listdir(root):
    try:
        directory = os.path.join(root, directory)
        run_id = directory[directory.rfind("-") + 1:]
        logger_config = load_toml(os.path.join(directory, "config.toml"))

        wandb.init(entity=logger_config.wandb.entity, project=logger_config.wandb.project_name, resume=run_id, reinit=True)

        config = yaml.load(open(os.path.join(directory, "config.yaml")))
        weights_path = config["results_path"]["value"]
        wandb.save(os.path.join(weights_path, "model.weights"))
    except Exception as e:
        print(e)
