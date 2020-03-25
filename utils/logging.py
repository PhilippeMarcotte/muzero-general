from abc import ABC

from comet_ml import Experiment
from torch.utils.tensorboard import SummaryWriter


class Logger(ABC):
    def __init__(self, config):
        self.config = config
        self.writer = None

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        pass

    def add_text(self, tag, text_string, global_step=None, walltime=None):
        pass


class TensorboardLogger(Logger):
    def __init__(self, config):
        super().__init__(config)
        self.writer = SummaryWriter(config.results_path)
        print(
            "\nTraining...\nRun tensorboard --logdir ./results and go to http://localhost:6006/ to see in real time the training performance.\n"
        )

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        self.writer.add_scalar(tag, scalar_value, global_step=None, walltime=None)

    def add_text(self, tag, text_string, global_step=None, walltime=None):
        self.writer.add_text(tag, text_string, global_step=None, walltime=None)


class CometMLLogger(Logger):
    def __init__(self, config):
        super().__init__(config)
        self.writer = Experiment(project_name=config.comet_ml.project_name, api_key=config.comet_ml.api_key)

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        self.writer.log_metric(name=tag, value=scalar_value, step=global_step, epoch=None)

    def add_text(self, tag, text_string, global_step=None, walltime=None):
        pass
