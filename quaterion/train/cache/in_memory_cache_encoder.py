from typing import Hashable, List

import torch

from torch import Tensor
from quaterion_models.encoders import Encoder
from quaterion_models.types import TensorInterchange, CollateFnType

from quaterion.train.cache.cache_config import CacheType
from quaterion.train.cache.cache_encoder import CacheEncoder


class InMemoryCacheEncoder(CacheEncoder):
    """CacheEncoder which is able to store tensors on CPU or GPU

    :meta private:
    """

    def __init__(
        self,
        encoder: Encoder,
        cache_type=CacheType.AUTO,
    ):
        super().__init__(encoder)
        self.cache = {}
        self._cache_type = cache_type
        self._original_device = None

    @property
    def cache_type(self):
        return self._cache_type

    def forward(self, batch: "TensorInterchange") -> Tensor:
        """Infer encoder - convert input batch to embeddings

        Args:
            batch: tuple of keys to retrieve values from cache

        Returns:
            Tensor: embeddings of shape [batch_size x embedding_size]
        """
        embeddings: torch.Tensor = torch.stack([self.cache[value] for value in batch])

        device = next(self.parameters(), torch.Tensor(0)).device
        if device != embeddings.device:
            embeddings = embeddings.to(device)

        return embeddings

    def get_collate_fn(self) -> "CollateFnType":
        """Provides function that converts raw data batch into suitable input.

        Returns:
            CacheCollateFnType: method that converts raw data batch into
                keys and encoder's input
        """
        return self.cache_collate

    def fill_cache(self, keys: List[Hashable], data: "TensorInterchange") -> None:
        embeddings = self._encoder(data)
        if self.cache_type == CacheType.CPU:
            embeddings = embeddings.to("cpu")
        if self.cache_type == CacheType.GPU:
            # ToDo: Move to which GPU?
            pass
        self.cache.update(dict(zip(keys, embeddings)))

    def reset_cache(self) -> None:
        """Resets cache."""
        self.cache.clear()
