import torch

from dataclasses import dataclass
from typing import Any, List

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
    @classmethod
    def collate_fn(cls, batch: List[SimilarityPairSample]):
        features = [record.obj_a for record in batch] \
                   + [record.obj_b for record in batch]
        labels = {
            "pairs": torch.LongTensor([[i, i + len(batch)] for i in range(len(batch))]),
            "labels": torch.Tensor([record.score for record in batch]),
        }
        return features, labels


class GroupSimilarityDataLoader(DataLoader[SimilarityGroupSample]):
    @classmethod
    def collate_fn(cls, batch: List[SimilarityGroupSample]):
        features = [record.obj for record in batch]
        labels = {
            "groups": torch.LongTensor([record.group for record in batch])
        }
        return features, labels
