"""
Implementations of various callbacks used throughout `Latte`, either in training VAEs or MIST.
"""

import copy
from collections import defaultdict

import pytorch_lightning as pl
from torch.utils.tensorboard import SummaryWriter

from pythae.trainers import training_callbacks

from latte.models.mine.mine import MINE, StandaloneMINE


class MetricTracker(pl.Callback):
    """A callback to track all the epoch-level metrics logged."""

    def __init__(self):
        self.epoch_outputs = []

    def on_validation_epoch_end(self, trainer: pl.Trainer, module: pl.LightningModule) -> None:
        self.epoch_outputs.append(copy.deepcopy(trainer.logged_metrics))


class WeightsTracker(pl.Callback):
    """A callback for keeping track of the weights throughout training.
    Used for unit testing."""

    def __init__(self):
        self.epoch_weights = defaultdict(list)

    def on_validation_epoch_end(self, trainer: pl.Trainer, module: pl.LightningModule) -> None:
        layers = (
            list(module.T.S)
            if (isinstance(module, MINE) or isinstance(module, StandaloneMINE))
            else list(module.mine.T.S) + [module.projection_layer_x.A]
        )
        # This only keeps track of the projection matrix of the first distribution in the case of MIST

        for ii, layer in enumerate(layers):
            if hasattr(layer, "weight"):
                self.epoch_weights[ii].append(layer.weight.cpu().detach().numpy().copy())


class TensorboardCallback(training_callbacks.TrainingCallback):
    """
    A `pythae` callback to log the train and eval metrics to `Tensorboard`.
    Useful when using the provided Weights and Biases callback is not possible.
    """

    def __init__(self):
        self.is_initialized = False
        self.summary_writer = None

    def on_train_begin(self, training_config, **kwargs):
        if not self.is_initialized:
            self.summary_writer = SummaryWriter()  # The writer to Tensorboard
            self.is_initialized = True

    def on_log(self, training_config, logs, **kwargs):
        global_step = kwargs.pop("global_step", None)

        # Rename the entries to be more suitable for Tensorboard by changing "_" into "/"
        logs = training_callbacks.rename_logs(logs)

        self.summary_writer.add_scalar("train/global_step", global_step, global_step)
        for key, val in logs.items():
            # Write out all the entries to Tensorboard
            self.summary_writer.add_scalar(key, val, global_step)
