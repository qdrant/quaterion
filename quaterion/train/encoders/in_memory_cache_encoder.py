from typing import Tuple, Hashable, Iterable

import torch

from torch import Tensor
from quaterion_models.encoders import Encoder
from quaterion_models.types import TensorInterchange

from quaterion.train.encoders.cache_config import CacheType
from quaterion.train.encoders.cache_encoder import (
    CacheEncoder,
    CacheCollateFnType,
    KeyExtractorType,
)


class InMemoryCacheEncoder(CacheEncoder):
    """CacheEncoder which is able to store tensors on CPU or GPU"""

    def __init__(
        self,
        encoder: Encoder,
        key_extractor: KeyExtractorType = None,
        cache_type=CacheType.AUTO,
    ):
        super().__init__(encoder, key_extractor)
        self.cache = {}
        self._cache_type = self.resolve_cache_type(cache_type)

    @staticmethod
    def resolve_cache_type(cache_type: CacheType) -> CacheType:
        """Resolve received cache type.

        If cache type is AUTO, then set it cuda if it is available, otherwise
        use cpu.

        Args:
            cache_type: cache type to be resolved

        Returns:
            CacheType
        """
        if cache_type == CacheType.AUTO:
            cache_type = CacheType.GPU if torch.cuda.is_available() else CacheType.CPU
        return cache_type

    @property
    def cache_type(self):
        return self._cache_type

    def forward(self, batch: TensorInterchange) -> Tensor:
        """Infer encoder - convert input batch to embeddings

        Args:
            batch: tuple of keys to retrieve values from cache

        Returns:
            Tensor: embeddings of shape [batch_size x embedding_size]
        """
        embeddings = torch.stack([self.cache[value] for value in batch])
        if self.cache_type == CacheType.CPU:
            device = next(self.parameters(), torch.Tensor(0)).device
            embeddings = embeddings.to(device)
        return embeddings

    def get_collate_fn(self) -> CacheCollateFnType:
        """Provides function that converts raw data batch into suitable input.

        Returns:
            CacheCollateFnType: method that converts raw data batch into
                keys and encoder's input
        """
        return self.cache_collate

    def fill_cache(self, data: Tuple[Iterable[Hashable], TensorInterchange]) -> None:
        """Apply wrapped encoder to data and store processed data on
        corresponding device.

        Args:
            data: Tuple of keys and batches suitable for encoder

        Returns:
            None
        """
        keys, batch = data
        embeddings = self._encoder(batch)
        if self.cache_type == CacheType.CPU:
            embeddings = embeddings.to("cpu")
        self.cache.update(dict(zip(keys, embeddings)))

    def reset_cache(self) -> None:
        """Resets cache.

        Returns:
            None
        """
        self.cache.clear()
        self.cache_filled = False
