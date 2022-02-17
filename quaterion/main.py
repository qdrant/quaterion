from functools import partial
from typing import Optional, Callable, List, Any, Tuple, Dict

import pytorch_lightning as pl

from torch import Tensor

from quaterion_models.types import TensorInterchange

from quaterion.dataset.similarity_data_loader import (
    PairsSimilarityDataLoader,
    GroupSimilarityDataLoader, SimilarityDataLoader,
)
from quaterion.loss import GroupLoss, PairwiseLoss
from quaterion.train.trainable_model import TrainableModel
from quaterion.types.batch import TrainBatch


class Quaterion:
    """A dwarf on a giant's shoulders sees farther of the two"""

    @classmethod
    def _combiner_collate_fn(
        cls,
        batch: List[Any],
        features_collate: Callable,
        labels_collate: Callable,
    ) -> Tuple[TensorInterchange, Dict[str, Tensor]]:
        """Combined collate function of dataloader and model's encoders.

        Args:
            batch: List of raw data
            features_collate: Model's collate function, it is responsible for converting
                extracted by `labels_collate` features into suitable model input.
            labels_collate: origin collate function of the data loader. It is responsible for
                extracting features and labels from raw data.

        Returns:
            Tuple[TensorInterchange, Dict[str, Tensor]]: Tuple of suitable model's input
                and labels
        """

        features, labels = labels_collate(batch)
        return features_collate(features), labels

    @classmethod
    def fit(
        cls,
        trainable_model: TrainableModel,
        trainer: pl.Trainer,
        train_dataloader: SimilarityDataLoader,
        val_dataloader: Optional[SimilarityDataLoader] = None,
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
        """

        if isinstance(train_dataloader, PairsSimilarityDataLoader):
            if isinstance(trainable_model.loss, PairwiseLoss):
                train_dataloader.collate_fn = partial(
                    cls._combiner_collate_fn,
                    features_collate=trainable_model.model.get_collate_fn(),
                    labels_collate=train_dataloader.__class__.pre_collate_fn,
                )
            elif isinstance(trainable_model.loss, GroupLoss):
                raise NotImplementedError(
                    "Can't use GroupLoss with PairsSimilarityDataLoader"
                )

        if isinstance(train_dataloader, GroupSimilarityDataLoader):
            if isinstance(trainable_model.loss, GroupLoss):
                train_dataloader.collate_fn = partial(
                    cls._combiner_collate_fn,
                    features_collate=trainable_model.model.get_collate_fn(),
                    labels_collate=train_dataloader.__class__.pre_collate_fn,
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
