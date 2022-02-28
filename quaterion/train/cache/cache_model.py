from typing import Dict, Optional

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
import torch

from quaterion.train.cache import CacheEncoder


class CacheModel(pl.LightningModule):
    """Mock model for convenient caching.

    This class is required to make caching process similar to the training of
    the genuine model and inherit and use the same trainer instance. It allows
    avoiding of messing with device managing stuff and more.

    Args:
        encoders: dict of cache encoders names and corresponding instances to cache

    :meta private:
    """

    def __init__(
        self,
        encoders: Dict[str, CacheEncoder],
    ):

        super().__init__()
        self.encoders = encoders
        for key, encoder in self.encoders.items():
            self.add_module(key, encoder)

    def predict_step(
        self,
        batch,
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ):
        """Caches batch of input.

        Args:
            batch: Tuple of feature batch and labels batch.
            batch_idx: Index of current batch
            dataloader_idx: Index of the current dataloader

        Returns:
            torch.Tensor: loss mock
        """
        features, _labels = batch
        for encoder_name, encoder in self.encoders.items():
            if encoder_name not in features:
                continue
            keys, encoder_features = features.get(encoder_name)
            if len(keys) == 0:
                # empty batch possible if all unique object already cached
                continue
            encoder.fill_cache(keys, encoder_features)

        return torch.Tensor([1])

    # region anchors
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        pass

    def test_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def val_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass

    # endregion
