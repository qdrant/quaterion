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
        return torch.stack([self.cache[value] for value in batch])

    def get_collate_fn(self) -> CollateFnType:
        """
        Provides function that converts raw data batch into suitable model
        input

        :return: Model input
        """
        return self.cache_encoder_collate

    def fill_cache(self, data: Collection[Hashable]) -> None:
        """
        Apply wrapped encoder to data and store it on gpu on which it was
        calculated

        :param data:
        :return: None
        """
        keys, batch = data
        embeddings = self._encoder(batch)
        self.cache.update(dict(zip(keys, embeddings)))

    def reset_cache(self) -> None:
        self.cache.clear()
        self.cache_filled = False
