from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, Any, Hashable


class CacheType(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    GPU = "gpu"


@dataclass
class CacheConfig:
    cache_type: Optional[CacheType] = None
    mapping: Dict[str, CacheType] = field(default_factory=dict)
    key_extractors: Dict[str, Callable[[Any], Hashable]] = field(
        default_factory=dict
    )
