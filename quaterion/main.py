from functools import partial
from typing import Optional, Callable, List, Any, Type

from torch.utils.data import DataLoader
from torch.optim import Optimizer, Adam

import pytorch_lightning as pl
from quaterion_models.model import MetricModel

from quaterion.dataset.similarity_data_loader import PairsSimilarityDataLoader, GroupSimilarityDataLoader
from quaterion.loss.group_loss import GroupLoss
from quaterion.loss.pairwise_loss import PairwiseLoss
from quaterion.loss.similarity_loss import SimilarityLoss
from quaterion.train.trainable_model import TrainableModel


class Quaterion:

    @classmethod
    def combiner_collate_fn(cls, batch: List[Any], features_collate: Callable, labels_collate: Callable):
        features, labels = labels_collate(batch)
        return features_collate(features), labels

    @classmethod
    def fit(cls,
            trainer: pl.Trainer,
            model: MetricModel,
            loss: SimilarityLoss,
            train_dataloader: DataLoader,
            val_dataloader: Optional[DataLoader],
            optimizer_class: Type[Optimizer] = Adam,
            optimizer_params: dict = None,
            trainable_model_class: Type[TrainableModel] = TrainableModel
            ):

        trainable_model = trainable_model_class(
            model=model,
            loss=loss,
            optimizer_class=optimizer_class,
            optimizer_params=optimizer_params
        )

        if isinstance(train_dataloader, PairsSimilarityDataLoader):
            if isinstance(loss, PairwiseLoss):
                train_dataloader.collate_fn = partial(
                    cls.combiner_collate_fn,
                    features_collate=model.get_collate_fn(),
                    labels_collate=train_dataloader.__class__.collate_fn
                )
            elif isinstance(loss, GroupLoss):
                raise NotImplementedError("Can't use GroupLoss with PairsSimilarityDataLoader")

        if isinstance(trainable_model, GroupSimilarityDataLoader):
            if isinstance(loss, GroupLoss):
                train_dataloader.collate_fn = partial(
                    cls.combiner_collate_fn,
                    features_collate=model.get_collate_fn(),
                    labels_collate=train_dataloader.__class__.collate_fn
                )

                train_dataloader.collate_fn = model.get_collate_fn()
            elif isinstance(loss, PairwiseLoss):
                raise NotImplementedError("Pair samplers are not implemented yet. Try other loss/data loader")

        if val_dataloader:
            val_dataloader.collate_fn = train_dataloader.collate_fn

        trainer.fit(
            model=trainable_model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )
