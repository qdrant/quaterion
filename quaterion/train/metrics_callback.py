from typing import Any

from loguru import logger
import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback

from quaterion.utils.enums import TrainStage


class MetricsCallback(Callback):
    @staticmethod
    def reset_estimators(trainable_model, stage=None):
        for estimator in trainable_model.estimators:
            if not stage or stage in estimator.name:
                estimator.reset()

    @staticmethod
    def reset_metrics(trainable_model):
        for metric in trainable_model.metrics:
            metric.reset()

    @staticmethod
    def log_and_reset_estimator(
        trainable_model, current_epoch, stage=None, last_epoch=False
    ):
        for estimator in trainable_model.estimators:
            if stage and stage not in estimator.name:
                continue

            if estimator.has_been_reset:
                continue

            if estimator.policy and current_epoch % estimator.policy == 0:
                trainable_model.log(
                    estimator.name, estimator.estimate(), logger=estimator.logger, prog_bar=True
                )

            if last_epoch:
                logger.info(f"{estimator.name} result: {estimator.estimate()}")
            else:
                estimator.reset()

    def on_sanity_check_end(self, trainer: "pl.Trainer", trainable_model: "pl.LightningModule") -> None:
        self.reset_estimators(trainable_model, TrainStage.VALIDATION)

    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        trainable_model: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        unused: int = 0,
    ) -> None:
        # warning: https://github.com/PyTorchLightning/pytorch-lightning/pull/12769
        self.reset_metrics(trainable_model)

    def on_validation_batch_start(
        self,
        trainer: "pl.Trainer",
        trainable_model: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self.reset_metrics(trainable_model)

    def on_test_epoch_start(
        self, trainer: "pl.Trainer", trainable_model: "pl.LightningModule"
    ) -> None:
        self.reset_metrics(trainable_model)

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", trainable_model: "pl.LightningModule"
    ) -> None:
        self.log_and_reset_estimator(
            trainable_model,
            trainer.current_epoch,
            stage=TrainStage.TRAIN,
            last_epoch=False,
        )

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", trainable_model: "pl.LightningModule"
    ) -> None:
        self.log_and_reset_estimator(
            trainable_model,
            trainer.current_epoch,
            stage=TrainStage.VALIDATION,
            last_epoch=False,
        )

    def on_test_epoch_end(
        self, trainer: "pl.Trainer", trainable_model: "pl.LightningModule"
    ) -> None:
        self.log_and_reset_estimator(
            trainable_model,
            trainer.current_epoch,
            stage=TrainStage.TEST,
            last_epoch=False,
        )

    def on_fit_end(
        self, trainer: "pl.Trainer", trainable_model: "pl.LightningModule"
    ) -> None:
        self.log_and_reset_estimator(
            trainable_model, trainer.current_epoch, last_epoch=True
        )

    def on_test_end(self, trainer: "pl.Trainer", trainable_model: "pl.LightningModule") -> None:
        self.log_and_reset_estimator(
            trainable_model, trainer.current_epoch, last_epoch=True
        )
