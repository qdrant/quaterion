from typing import Collection, Hashable, Any, List, Callable, Union, Tuple

from torch import Tensor

from quaterion_models.encoders import Encoder
from quaterion_models.types import TensorInterchange

KeyExtractorType = Callable[[Any], Hashable]

CacheCollateFnType = Callable[
    [Collection[Any]],
    Union[List[Hashable], Tuple[Hashable, TensorInterchange]],
]


class CacheEncoder(Encoder):
    def __init__(self, encoder: Encoder, key_extractor: KeyExtractorType = None):
        if encoder.trainable():
            raise ValueError("Trainable encoder can't be cached")
        super().__init__()
        self._encoder = encoder
        self.cache_filled = False
        self.key_extractor = (
            key_extractor if key_extractor is not None else self.default_key_extractor
        )

    def trainable(self) -> bool:
        return False

    def embedding_size(self) -> int:
        """
        :return: Size of resulting embedding
        """
        return self._encoder.embedding_size()

    @classmethod
    def default_key_extractor(cls, obj: Any) -> Hashable:
        """
        Generate hashable from batch object
        :return: Key for cache
        """
        return (
            hash(obj) if not isinstance(obj, dict) else hash(tuple(sorted(obj.items())))
        )

    def key_collate_fn(self, batch: Collection[Any]) -> List[Hashable]:
        """
        Convert batch of raw data into list of keys for cache
        :return: List of keys for cache
        """
        return [self.key_extractor(value) for value in batch]

    def cache_collate(
        self, batch: Collection[Any]
    ) -> Union[List[Hashable], Tuple[Hashable, TensorInterchange]]:
        """
        Converts raw data batch into suitable model input and keys for caching

        :return: Tuple of keys and according model input
        """
        keys: List[Hashable] = self.key_collate_fn(batch)
        if self.cache_filled:
            return keys
        values: TensorInterchange = self._encoder.get_collate_fn()(batch)
        return keys, values

    def get_collate_fn(self) -> CacheCollateFnType:
        """
        Provides function that converts raw data batch into suitable model
        input

        :return: Model input
        """
        return self.cache_collate

    def forward(self, batch: TensorInterchange) -> Tensor:
        """
        Infer encoder - convert input batch to embeddings

        :param batch: processed batch
        :return: embeddings, shape: [batch_size x embedding_size]
        """
        raise NotImplementedError()

    def save(self, output_path: str):
        """
        Persist current state to the provided directory

        :param output_path:
        :return:
        """
        self._encoder.save(output_path)

    @classmethod
    def load(cls, input_path: str) -> "Encoder":
        """
        CachedEncoder classes wrap already instantiated encoders and don't
        provide loading support.

        :param input_path:
        :return:
        """
        raise ValueError("Cached encoder does not support loading")

    def fill_cache(
        self,
        data: Collection[Hashable],
    ):
        """
        Applies encoder to data and store results in cache

        :param data: collection of hashables to which encoder will be applied
        and the resulting embeddings will be stored in cache
        """
        raise NotImplementedError()

    def reset_cache(self):
        """
        Reset all stored data
        """
        raise NotImplementedError()
