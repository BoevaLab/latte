import pytorch_lightning as pl


# TODO (Anej)
class MetricTracker(pl.Callback):
    def __init__(self):
        self.collection = []

    def on_validation_batch_end(self, trainer, module, outputs):
        vacc = outputs["val_acc"]
        self.collection.append(vacc)

    def on_validation_epoch_end(self, trainer, module):
        elogs = trainer.logged_metrics
        self.collection.append(elogs)
