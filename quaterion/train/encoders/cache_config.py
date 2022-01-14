from enum import IntEnum
from dataclasses import dataclass
from typing import Dict, Optional


class CacheType(IntEnum):
    AUTO = 1
    CPU = 2
    GPU = 3


@dataclass
class CacheConfig:
    cache_type: Optional[CacheType] = None
    mapping: Optional[Dict[str, CacheType]] = None
