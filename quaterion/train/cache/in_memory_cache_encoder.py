import pickle
from typing import Any, Hashable, List, Union

import torch
from quaterion_models.encoders import Encoder
from quaterion_models.types import CollateFnType, TensorInterchange
from torch import Tensor

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
        self._cache = None
        self._meta_cache = {}
        self._offset_map = {}
        self._cache_type = cache_type
        self._original_device = None
        self._tmp = []

    def _encoder_device(self):
        return next(self._encoder.parameters(), torch.Tensor(0)).device

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
        offsets = [self._offset_map[key] for key in batch]
        embeddings: torch.Tensor = self._cache[offsets]

        device = self._encoder_device()
        if device != embeddings.device:
            embeddings = embeddings.to(device)

        return embeddings

    def cache_extract_meta(self, batch: Union[tuple, List[Hashable]]) -> List[dict]:
        if isinstance(batch, tuple):
            # Cache filling phase
            _keys, features = batch
            return self._encoder.get_meta_extractor()(features)
        else:
            # Assume training phase.
            # Only keys are provided here
            return [self._meta_cache[key] for key in batch]

    def get_collate_fn(self) -> "CollateFnType":
        """Provides function that converts raw data batch into suitable input.

        Returns:
            CacheCollateFnType: method that converts raw data batch into
                keys and encoder's input
        """
        return self.cache_collate

    def is_filled(self) -> bool:
        return self._cache is not None

    def fill_cache(
        self, keys: List[Hashable], data: "TensorInterchange", meta: List[Any]
    ) -> None:
        embeddings = self._encoder(data)
        if self.cache_type == CacheType.CPU:
            embeddings = embeddings.to("cpu")
        if self.cache_type == CacheType.GPU:
            # ToDo: Move to which GPU?
            pass
        for key in keys:
            self._offset_map[key] = len(self._offset_map)
        self._tmp.append(embeddings)

        for key, meta_item in zip(keys, meta):
            self._meta_cache[key] = meta_item

    def finish_fill(self):
        self._cache = torch.cat(self._tmp)
        self._tmp = []

    def reset_cache(self) -> None:
        """Resets cache."""
        self._cache = None
        self._meta_cache = {}
        self._offset_map = {}
        self._tmp = []

    def save_cache(self, path):
        pickle.dump(
            [self._cache.to("cpu"), self._offset_map, self._meta_cache],
            open(path, "wb"),
        )

    def load_cache(self, path):
        self._cache, self._offset_map, self._meta_cache = pickle.load(open(path, "rb"))
        if self.cache_type != CacheType.CPU:
            device = self._encoder_device()
            self._cache = self._cache.to(device)
