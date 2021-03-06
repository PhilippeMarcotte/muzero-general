from abc import ABC

import wandb
from torch.utils.tensorboard import SummaryWriter


class Logger(ABC):
    def __init__(self, logger_config, config):
        self.logger_config = logger_config
        self.config = config
        self.writer = None

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        pass

    def add_text(self, tag, text_string, global_step=None, walltime=None):
        pass


class TensorboardLogger(Logger):
    def __init__(self, logger_config, config):
        super().__init__(logger_config, config)
        self.writer = SummaryWriter(config.results_path)
        print(
            "\nTraining...\nRun tensorboard --logdir ./results and go to http://localhost:6006/ to see in real time the training performance.\n"
        )

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        self.writer.add_scalar(tag, scalar_value, global_step=None, walltime=None)

    def add_text(self, tag, text_string, global_step=None, walltime=None):
        self.writer.add_text(tag, text_string, global_step=None, walltime=None)


class WandbLogger(Logger):
    def __init__(self, logger_config, config, tags):
        super().__init__(logger_config, config)
        self.writer = wandb
        wandb.init(project=logger_config.wandb.project_name, entity=logger_config.wandb.entity, group=logger_config.wandb.group, config=config, tags=tags)

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        self.writer.log(row={tag: scalar_value}, commit=None, step=global_step, sync=True)

    def add_text(self, tag, text_string, global_step=None, walltime=None):
        pass
