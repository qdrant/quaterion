from typing import Any

from loguru import logger
import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback

from quaterion.utils.enums import TrainStage


class MetricsCallback(Callback):
    @staticmethod
    def reset_evaluators(trainable_model, stage=None):
        for evaluator in trainable_model.evaluators:
            if not stage or stage in evaluator.name:
                evaluator.reset()

    @staticmethod
    def reset_metrics(trainable_model):
        for metric in trainable_model.metrics:
            metric.reset()

    @staticmethod
    def log_and_reset_evaluator(
        trainable_model, current_epoch, stage=None, last_epoch=False
    ):
        for evaluator in trainable_model.evaluators:
            if stage and stage not in evaluator.name:
                continue

            if evaluator.has_been_reset:
                continue

            if evaluator.policy and current_epoch % evaluator.policy == 0:
                trainable_model.log(
                    evaluator.name,
                    evaluator.estimate(),
                    logger=evaluator.logger,
                    prog_bar=True,
                )

            if last_epoch:
                logger.info(f"{evaluator.name} result: {evaluator.estimate()}")
            else:
                evaluator.reset()

    def on_sanity_check_end(
        self, trainer: "pl.Trainer", trainable_model: "pl.LightningModule"
    ) -> None:
        self.reset_evaluators(trainable_model, TrainStage.VALIDATION)

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
        self.log_and_reset_evaluator(
            trainable_model,
            trainer.current_epoch,
            stage=TrainStage.TRAIN,
            last_epoch=False,
        )

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", trainable_model: "pl.LightningModule"
    ) -> None:
        self.log_and_reset_evaluator(
            trainable_model,
            trainer.current_epoch,
            stage=TrainStage.VALIDATION,
            last_epoch=False,
        )

    def on_test_epoch_end(
        self, trainer: "pl.Trainer", trainable_model: "pl.LightningModule"
    ) -> None:
        self.log_and_reset_evaluator(
            trainable_model,
            trainer.current_epoch,
            stage=TrainStage.TEST,
            last_epoch=False,
        )

    def on_fit_end(
        self, trainer: "pl.Trainer", trainable_model: "pl.LightningModule"
    ) -> None:
        self.log_and_reset_evaluator(
            trainable_model, trainer.current_epoch, last_epoch=True
        )

    def on_test_end(
        self, trainer: "pl.Trainer", trainable_model: "pl.LightningModule"
    ) -> None:
        self.log_and_reset_evaluator(
            trainable_model, trainer.current_epoch, last_epoch=True
        )
