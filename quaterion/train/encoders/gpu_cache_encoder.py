from typing import Hashable, Collection

import torch
from torch import Tensor

from quaterion_models.encoder import Encoder, CollateFnType

from quaterion.train.encoders.cache_encoder import CacheEncoder


class GpuCacheEncoder(CacheEncoder):
    def __init__(self, encoder: Encoder):
        super().__init__(encoder)
        self.cache = {}

    def forward(self, batch: Collection[Hashable]) -> Tensor:
        """
        Infer encoder - convert input batch to embeddings

        :param batch: processed batch
        :return: embeddings, shape: [batch_size x embedding_size]
        """
        return torch.stack([self.cache[hash(value)] for value in batch])

    def dummy_collate(
        self, batch: Collection[Hashable]
    ) -> Collection[Hashable]:
        """
        Collate function designed for proper cache usage

        :param batch:
        :return: Collection[Hashable]
        """
        return batch

    def get_collate_fn(self) -> CollateFnType:
        return self.dummy_collate

    def fill_cache(self, data: Collection[Hashable]) -> None:
        """
        Apply wrapped encoder to data and store it on gpu on which it was
        calculated

        :param data:
        :return: None
        """
        inner_collate_fn = self._encoder.get_collate_fn()
        embeddings = self._encoder(inner_collate_fn(data))
        hashes = (hash(obj) for obj in data)
        self.cache.update(dict(zip(hashes, embeddings)))

    def reset_cache(self) -> None:
        self.cache = {}
