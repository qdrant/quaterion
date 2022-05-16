import torch
from typing import Optional, Union, Sized, Iterable, Dict

import pytorch_lightning as pl
from torch.utils.data import Dataset
from quaterion_models import SimilarityModel

from quaterion.dataset.similarity_data_loader import (
    PairsSimilarityDataLoader,
    GroupSimilarityDataLoader,
    SimilarityDataLoader,
)
from quaterion.eval.evaluator import Evaluator
from quaterion.loss import GroupLoss, PairwiseLoss
from quaterion.train.cleanup_callback import CleanupCallback
from quaterion.train.metrics_callback import MetricsCallback
from quaterion.train.trainable_model import TrainableModel


class Quaterion:
    """A dwarf on a giant's shoulders sees farther of the two"""

    @classmethod
    def fit(
        cls,
        trainable_model: TrainableModel,
        trainer: pl.Trainer,
        train_dataloader: SimilarityDataLoader,
        val_dataloader: Optional[SimilarityDataLoader] = None,
        ckpt_path: Optional[str] = None,
    ):
        """Handle training routine

        Assemble data loaders, performs caching and whole training process.

        Args:
            trainable_model: model to fit
            trainer: `pytorch_lightning.Trainer` instance to handle fitting routine
                internally
            train_dataloader: DataLoader instance to retrieve samples during training
                stage
            val_dataloader: Optional DataLoader instance to retrieve samples during
                validation stage
            ckpt_path: Path/URL of the checkpoint from which training is resumed. If there is
                no checkpoint file at the path, an exception is raised. If resuming from mid-epoch checkpoint,
                training will start from the beginning of the next epoch.
        """

        if isinstance(train_dataloader, PairsSimilarityDataLoader):
            if not isinstance(trainable_model.loss, PairwiseLoss):
                raise NotImplementedError(
                    "Can't use PairsSimilarityDataLoader with non-PairwiseLoss"
                )

        if isinstance(train_dataloader, GroupSimilarityDataLoader):
            if not isinstance(trainable_model.loss, GroupLoss):
                raise NotImplementedError(
                    "Pair samplers are not implemented yet. "
                    "Try other loss/data loader"
                )

        trainer.callbacks.append(CleanupCallback())
        trainer.callbacks.append(MetricsCallback())
        # Prepare data loaders for training

        trainable_model.setup_dataloader(train_dataloader)
        if val_dataloader:
            trainable_model.setup_dataloader(val_dataloader)

        trainable_model.setup_cache(
            trainer=trainer,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
        )

        trainer.fit(
            model=trainable_model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=ckpt_path,
        )

    @classmethod
    def evaluate(
        cls,
        evaluator: Evaluator,
        dataset: Union[Sized, Iterable, Dataset],
        model: SimilarityModel,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute metrics on a dataset

        Args:
            evaluator: Object which holds the configuration of which metrics to use and how to obtain samples for them
            dataset: Sized object, like list, tuple, torch.utils.data.Dataset, etc. to compute metrics
            model: SimilarityModel instance to perform objects encoding

        Returns:
            Dict[str, torch.Tensor] - dict of computed metrics.
            Where key - name of the metric and value - metric estimated values

        """
        return evaluator.evaluate(dataset, model)
