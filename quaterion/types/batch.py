from dataclasses import dataclass
from typing import List, Any


@dataclass
class TrainBatch:
    # Input data for encoders
    features: List[Any]
    # # Input hash identifiers, based on hash of sequential order and loading params
    # index_hash: List[int]
