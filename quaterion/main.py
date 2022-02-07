from functools import partial
from typing import Optional, Callable, List, Any

import pytorch_lightning as pl

from torch.utils.data import DataLoader

from quaterion.dataset.similarity_data_loader import (
    PairsSimilarityDataLoader,
    GroupSimilarityDataLoader,
)
from quaterion.loss import GroupLoss, PairwiseLoss
from quaterion.train.trainable_model import TrainableModel


class Quaterion:
    @classmethod
    def combiner_collate_fn(
        cls,
        batch: List[Any],
        features_collate: Callable,
        labels_collate: Callable,
    ):
        features, labels = labels_collate(batch)
        return features_collate(features), labels

    @classmethod
    def fit(
        cls,
        trainable_model: TrainableModel,
        trainer: pl.Trainer,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
    ):

        if isinstance(train_dataloader, PairsSimilarityDataLoader):
            if isinstance(trainable_model.loss, PairwiseLoss):
                train_dataloader.collate_fn = partial(
                    cls.combiner_collate_fn,
                    features_collate=trainable_model.model.get_collate_fn(),
                    labels_collate=train_dataloader.__class__.collate_fn,
                )
            elif isinstance(trainable_model.loss, GroupLoss):
                raise NotImplementedError(
                    "Can't use GroupLoss with PairsSimilarityDataLoader"
                )

        if isinstance(train_dataloader, GroupSimilarityDataLoader):
            if isinstance(trainable_model.loss, GroupLoss):
                train_dataloader.collate_fn = partial(
                    cls.combiner_collate_fn,
                    features_collate=trainable_model.model.get_collate_fn(),
                    labels_collate=train_dataloader.__class__.collate_fn,
                )
            elif isinstance(trainable_model.loss, PairwiseLoss):
                raise NotImplementedError(
                    "Pair samplers are not implemented yet. "
                    "Try other loss/data loader"
                )

        if val_dataloader is not None:
            val_dataloader.collate_fn = train_dataloader.collate_fn

        trainable_model.cache(
            trainer,
            trainable_model.model.encoders,
            train_dataloader,
            val_dataloader,
            trainable_model.cache_config,
        )

        trainer.fit(
            model=trainable_model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
