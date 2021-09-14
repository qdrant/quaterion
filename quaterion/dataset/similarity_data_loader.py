from dataclasses import dataclass
from typing import Any

from torch.utils.data import DataLoader


@dataclass
class SimilarityPairSample:
    obj_a: Any
    obj_b: Any
    score: float


@dataclass
class SimilarityGroupSample:
    obj: Any
    group: int


class PairsSimilarityDataLoader(DataLoader[SimilarityPairSample]):
    pass


class GroupSimilarityDataLoader(DataLoader[SimilarityGroupSample]):
    pass
