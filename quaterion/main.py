from typing import Optional

import pytorch_lightning as pl

from quaterion.dataset.similarity_data_loader import (
    PairsSimilarityDataLoader,
    GroupSimilarityDataLoader,
    SimilarityDataLoader,
)
from quaterion.dataset.train_collater import TrainCollater
from quaterion.loss import GroupLoss, PairwiseLoss
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

        encoder_collate_fns = dict(
            (key, encoder.get_collate_fn())
            for key, encoder in trainable_model.model.encoders.items()
        )

        collater = TrainCollater(
            pre_collate_fn=train_dataloader.collate_fn,
            encoder_collates=encoder_collate_fns
        )

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

        train_dataloader.collate_fn = collater
        if val_dataloader is not None:
            val_dataloader.collate_fn = collater

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
