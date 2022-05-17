import copy
from collections import defaultdict

import pytorch_lightning as pl

from latte.models import mine


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

    def on_validation_epoch_end(self, trainer: pl.Trainer, module: mine.MINE) -> None:
        for ii, layer in enumerate(module.T.S):
            if hasattr(layer, "weight"):
                self.epoch_weights[ii].append(copy.deepcopy(layer.weight))
