from enum import Enum
from typing import Hashable, Any, List, Callable, Union, Tuple

from torch import Tensor

from quaterion_models.encoders import Encoder
from quaterion_models.types import TensorInterchange, CollateFnType

KeyExtractorType = Callable[[Any], Hashable]

CacheCollateReturnType = Union[
    List[Hashable], Tuple[List[Hashable], "TensorInterchange"]
]


class CacheMode(str, Enum):
    FILL = "fill"
    TRAIN = "train"


class CacheEncoder(Encoder):
    """Wrapper for encoders to avoid repeated calculations.

    Encoder results can be calculated one time and reused after in situations when
    encoder's layers are frozen and provide deterministic embeddings for input data.

    Args:
        encoder: Encoder object to be wrapped.

    :meta private:
    """

    def __init__(self, encoder: Encoder):
        if encoder.trainable():
            raise ValueError("Trainable encoder can't be cached")
        super().__init__()
        self._encoder = encoder

    @property
    def wrapped_encoder(self):
        return self._encoder

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

    def cache_collate(
        self, batch: Union[Tuple[List[Hashable], List[Any]], List[Hashable]]
    ) -> "CacheCollateReturnType":
        """Converts raw data batch into suitable model input and keys for caching.

        Returns:
            In case only cache keys are provided: return keys
            If keys and actual features are provided -
                return result of original collate along with cache keys
        """
        if isinstance(batch, tuple):
            # Cache filling phase
            keys, features = batch
            collated_features = self._encoder.get_collate_fn()(features)
            return keys, collated_features
        else:
            # Assume training phase.
            # Only keys are provided here
            return batch

    def get_collate_fn(self) -> "CollateFnType":
        """Provides function that converts raw data batch into suitable model input.

        Returns:
             CacheCollateFnType: cache collate function
        """
        return self.cache_collate

    def forward(self, batch: "TensorInterchange") -> Tensor:
        """Infer encoder.

        Convert input batch to embeddings

        Args:
            batch: collated batch (currently, it can be only batch of keys)
        Returns:
            Tensor: shape: (batch_size, embedding_size) - embeddings
        """
        raise NotImplementedError()

    def save(self, output_path: str) -> None:
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

    def fill_cache(self, keys: List[Hashable], data: "TensorInterchange") -> None:
        """Apply wrapped encoder to data and store processed data on
        corresponding device.

        Args:
            keys: Hash keys which should be associated with resulting vectors
            data: Tuple of keys and batches suitable for encoder

        """
        raise NotImplementedError()

    def reset_cache(self):
        """Reset all stored data."""
        raise NotImplementedError()
