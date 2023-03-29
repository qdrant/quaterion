import pytorch_lightning as pl

try:  # fix for version >= 1.9.0
    from pytorch_lightning import Callback
except ImportError:
    from pytorch_lightning.callbacks.base import Callback


class MetricsCallback(Callback):
    @staticmethod
    def reset_metrics(trainable_model):
        for metric in trainable_model.attached_metrics:
            metric.reset()

    def on_sanity_check_end(
        self, trainer: "pl.Trainer", trainable_model: "pl.LightningModule"
    ) -> None:
        self.reset_metrics(trainable_model)

    def on_train_epoch_start(
        self, trainer: "pl.Trainer", trainable_model: "pl.LightningModule"
    ) -> None:
        self.reset_metrics(trainable_model)

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", trainable_model: "pl.LightningModule"
    ) -> None:
        self.reset_metrics(trainable_model)

    def on_validation_epoch_start(
        self, trainer: "pl.Trainer", trainable_model: "pl.LightningModule"
    ) -> None:
        self.reset_metrics(trainable_model)

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", trainable_model: "pl.LightningModule"
    ) -> None:
        self.reset_metrics(trainable_model)
