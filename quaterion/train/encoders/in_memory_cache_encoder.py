from typing import Tuple, Union

import torch
from torch import Tensor
from quaterion_models.encoders import Encoder
from quaterion_models.types import TensorInterchange

from quaterion.train.encoders.cache_config import CacheType
from quaterion.train.encoders.cache_encoder import CacheEncoder, CacheCollateFnType, KeyExtractorType


class InMemoryCacheEncoder(CacheEncoder):
    def __init__(self, encoder: Encoder, key_extractor: KeyExtractorType = None, cache_type=CacheType.AUTO):
        super().__init__(encoder, key_extractor)
        self.cache = {}
        self._cache_type = self.resolve_cache_type(cache_type)

    @staticmethod
    def resolve_cache_type(cache_type: CacheType) -> CacheType:
        if cache_type == CacheType.AUTO:
            cache_type = CacheType.GPU if torch.cuda.is_available() else CacheType.CPU
        return cache_type

    @property
    def cache_type(self):
        return self._cache_type

    def forward(self, batch: TensorInterchange) -> Tensor:
        """
        Infer encoder - convert input batch to embeddings

        :param batch: processed batch
        :return: embeddings, shape: [batch_size x embedding_size]
        """
        embeddings = torch.stack([self.cache[value] for value in batch])
        if self.cache_type == CacheType.CPU:
            device = next(self.parameters(), torch.Tensor(0)).device
            embeddings = embeddings.to(device)
        return embeddings

    def get_collate_fn(self) -> CacheCollateFnType:
        """
        Provides function that converts raw data batch into suitable model
        input

        :return: Model input
        """
        return self.cache_collate

    def fill_cache(self, data: Tuple[Union[str, int], TensorInterchange]) -> None:
        """
        Apply wrapped encoder to data and store it on corresponding device

        :param data: keys for mapping and batch of data to be passed to encoder
        :return: None
        """
        keys, batch = data
        embeddings = self._encoder(batch)
        if self.cache_type == CacheType.CPU:
            embeddings = embeddings.to("cpu")
        self.cache.update(dict(zip(keys, embeddings)))

    def reset_cache(self) -> None:
        self.cache.clear()
        self.cache_filled = False
