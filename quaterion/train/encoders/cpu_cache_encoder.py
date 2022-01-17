from typing import Tuple, Union

import torch
from torch import Tensor
from quaterion_models.encoder import Encoder, TensorInterchange, CollateFnType

from quaterion.train.encoders.cache_encoder import CacheEncoder


class CpuCacheEncoder(CacheEncoder):
    def __init__(self, encoder: Encoder):
        super().__init__(encoder)
        self.cache = {}

    def forward(self, batch: TensorInterchange) -> Tensor:
        """
        Infer encoder - convert input batch to embeddings

        :param batch: processed batch
        :return: embeddings, shape: [batch_size x embedding_size]
        """
        device = next(self.parameters(), torch.Tensor(0)).device
        return torch.stack([self.cache[value] for value in batch]).to(device)

    def get_collate_fn(self) -> CollateFnType:
        """
        Provides function that converts raw data batch into suitable model
        input

        :return: Model input
        """
        return self.cache_collate

    def fill_cache(
        self, data: Tuple[Union[str, int], TensorInterchange]
    ) -> None:
        """
        Apply wrapped encoder to data and store it on cpu

        Data being split into batches of batch size to accelerate encoding

        :param data: keys for mapping and batch of data to be passed to encoder
        :return: None
        """
        keys, batch = data
        embeddings = self._encoder(batch).to("cpu")
        self.cache.update(dict(zip(keys, embeddings)))

    def reset_cache(self) -> None:
        self.cache.clear()
        self.cache_filled = False
