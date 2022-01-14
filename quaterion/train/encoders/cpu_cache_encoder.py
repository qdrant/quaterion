from math import ceil
from typing import Hashable, Collection

import torch
from torch import Tensor
from tqdm import trange
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

    def fill_cache(
        self,
        data: Collection[Hashable],
        batch_size: int = 128,
        verbose: bool = False,
    ) -> None:
        """
        Apply wrapped encoder to data and store it on cpu

        Data being split into batches of batch size to accelerate encoding

        :param data:
        :param batch_size:
        :param verbose: enable/disable tqdm progress bar
        :return: None
        """
        inner_collate_fn = self._encoder.get_collate_fn()

        embeddings = []

        number_of_batches = ceil(len(data) / batch_size)
        batch_nums = (
            trange(number_of_batches) if verbose else range(number_of_batches)
        )
        for batch_num in batch_nums:
            start, end = batch_num * batch_size, (batch_num + 1) * batch_size
            batch = self._encoder(inner_collate_fn(data[start:end])).to("cpu")
            embeddings.append(batch)

        embeddings = torch.cat(embeddings)

        self.cache.update(dict(zip(data, embeddings)))

    def reset_cache(self) -> None:
        self.cache.clear()
