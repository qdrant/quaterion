from typing import Collection, Hashable, Any, List, Callable, Union, Tuple, Iterable

from torch import Tensor

from quaterion_models.encoders import Encoder
from quaterion_models.types import TensorInterchange


KeyExtractorType = Callable[[Any], Hashable]

CacheCollateReturnType = Union[List[Hashable], Tuple[Hashable, TensorInterchange]]

CacheCollateFnType = Callable[
    [Collection[Any]],
    CacheCollateReturnType,
]


class CacheEncoder(Encoder):
    """Wrapper for encoders to avoid repeated calculations.

    Encoder results can be calculated one time and reused after in situations when
    encoder's layers are frozen and provide deterministic embeddings for input data.

    Args:
        encoder: Encoder object to be wrapped.
        key_extractor: function required to extract hashable key from complicated
            objects which can't be hashed with default key extractor.
    """

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
        """Defines if encoder is trainable. This flag affects caching and checkpoint
        saving of the encoder.

        Returns:
            bool: whether encoder trainable or not
        """
        return False

    def embedding_size(self) -> int:
        """Size of output embedding.

        Returns:
            int: Size of resulting embedding.
        """
        return self._encoder.embedding_size()

    @classmethod
    def default_key_extractor(cls, obj: Any) -> Hashable:
        """Default implementation of key extractor.

        Generate hashable from batch object with built-in hash, also support dicts.

        Returns:
             Hashable: Key for cache
        """
        return hash(obj) if not isinstance(obj, dict) else hash(tuple(sorted(obj.items())))

    def key_collate_fn(self, batch: Collection[Any]) -> List[Hashable]:
        """Convert batch of raw data into list of keys for cache.

        Returns:
             List[Hashable]: List of keys for cache
        """
        return [self.key_extractor(value) for value in batch]

    def cache_collate(
        self, batch: Collection[Any]
    ) -> Union[List[Hashable], Tuple[Hashable, TensorInterchange]]:
        """Converts raw data batch into suitable model input and keys for caching.

        Returns:
             Union[
                List[Hashable],
                Tuple[Hashable, TensorInterchange]
            ]: Tuple of keys and according model input during caching process, only
            list of keys after cache has been filled
        """
        keys: List[Hashable] = self.key_collate_fn(batch)
        if self.cache_filled:
            return keys
        values: TensorInterchange = self._encoder.get_collate_fn()(batch)
        return keys, values

    def get_collate_fn(self) -> CacheCollateFnType:
        """Provides function that converts raw data batch into suitable model input.

        Returns:
             CacheCollateFnType: cache collate function
        """
        return self.cache_collate

    def forward(self, batch: TensorInterchange) -> Tensor:
        """Infer encoder.

        Convert input batch to embeddings

        Args:
            batch: collated batch (currently, it can be only batch of keys)
        Returns:
            Tensor: shape: (batch_size, embedding_size) - embeddings
        """
        raise NotImplementedError()

    def save(self, output_path: str):
        """Persist current state to the provided directory

        Args:
            output_path: path to save to
        """
        self._encoder.save(output_path)

    @classmethod
    def load(cls, input_path: str) -> Encoder:
        """CachedEncoder classes wrap already instantiated encoders and don't
        provide loading support.

        Args:
            input_path: path to load from
        """
        raise ValueError("Cached encoder does not support loading")

    def fill_cache(self, data: Tuple[Iterable[Hashable], TensorInterchange]) -> None:
        """Apply wrapped encoder to data and store processed data on
        corresponding device.

        Args:
            data: Tuple of keys and batches suitable for encoder

        """
        raise NotImplementedError()

    def reset_cache(self):
        """Reset all stored data."""
        raise NotImplementedError()
