from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, Any, Hashable, Union

#: Type of function to extract hash value from the input object.
#: Required if there is no other way to distinguish values for caching
KeyExtractorType = Callable[[Any], Hashable]


class CacheType(str, Enum):
    """Available tensor devices to be used for caching."""

    AUTO = "auto"
    """Use CUDA if it is available, else use CPU."""

    CPU = "cpu"
    """Tensors device is CPU."""

    GPU = "gpu"
    """Tensors device is GPU."""


@dataclass
class CacheConfig:
    """Class to be used in
    :meth:`~quaterion.train.trainable_model.TrainableModel.configure_caches`
    """

    cache_type: Optional[CacheType] = CacheType.AUTO
    """Cache type used for cacheable encoders not set in mapping"""

    mapping: Dict[str, CacheType] = field(default_factory=dict)
    """Mapping of `encoder_name` to :class:`~CacheType`"""

    key_extractors: Union[KeyExtractorType, Dict[str, KeyExtractorType]] = field(
        default_factory=dict
    )
    """Mapping of encoders to key extractor functions required to cache non-hashable 
    objects."""

    batch_size: Optional[int] = 32
    """Batch size to be used in CacheDataLoader during caching process. It does not 
    affect others training stages."""

    num_workers: Optional[int] = None  # if None - inherited from source dl
    """Num of workers to be used in CacheDataLoader during caching process. It does not 
    affect others training stages."""
