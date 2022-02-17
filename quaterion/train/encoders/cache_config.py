from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, Any, Hashable

# Function to extract hash value from the input object
# Required if there is no other way to distinguish values for caching
KeyExtractorType = Callable[[Any], Hashable]


class CacheType(str, Enum):
    """Available tensor devices to be used for caching.

    AUTO: use CUDA if it is available, else use CPU.
    CPU: tensors device is CPU.
    GPU: tensors device is GPU.
    """

    AUTO = "auto"
    CPU = "cpu"
    GPU = "gpu"


@dataclass
class CacheConfig:
    """Class to be used in `configure_cache` of `TrainableModel`.

    cache_type: cache type for single encoder which has no custom name set. In all other
        cases use `mapping`.
    mapping: mapping of `encoder_name` to `CacheType`, only specified inhere encoders
        will be cached, the others will stay untouched.
    key_extractors: mapping of encoders to key extractor functions required to cache
        non-hashable objects.
    batch_size: batch size to be used in CacheDataLoader during caching process. It does
        not affect others training stages.
    num_workers: num_workers to be used in CacheDataLoader during caching process. It
        does not affect others training stages.
    """

    cache_type: Optional[CacheType] = None
    batch_size: Optional[int] = 32
    num_workers: Optional[int] = None  # if None - inherited from source dl
    key_extractors: Dict[str, KeyExtractorType] = field(default_factory=dict)
    mapping: Dict[str, CacheType] = field(default_factory=dict)
