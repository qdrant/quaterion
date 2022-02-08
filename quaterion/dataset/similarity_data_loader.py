from typing import Any, List, Generic

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import T_co

from quaterion.dataset.similarity_samples import (
    SimilarityPairSample,
    SimilarityGroupSample,
)


class SimilarityDataLoader(DataLoader, Generic[T_co]):
    @classmethod
    def fetch_unique_objects(cls, batch: List[Any]) -> List[Any]:
        """
        Fetch unique objects from batch to avoid calculation of embeddings
        to repeated objects
        """
        raise NotImplementedError()


class PairsSimilarityDataLoader(SimilarityDataLoader[SimilarityPairSample]):
    @classmethod
    def collate_fn(cls, batch: List[SimilarityPairSample]):
        features = [record.obj_a for record in batch] + [
            record.obj_b for record in batch
        ]
        labels = {
            "pairs": torch.LongTensor([[i, i + len(batch)] for i in range(len(batch))]),
            "labels": torch.Tensor([record.score for record in batch]),
            "subgroups": torch.Tensor([record.subgroup for record in batch] * 2),
        }
        return features, labels

    @classmethod
    def fetch_unique_objects(cls, batch: List[SimilarityPairSample]):
        unique_objects = []
        for sample in batch:
            if sample.obj_a not in unique_objects:
                unique_objects.append(sample.obj_a)
            if sample.obj_b not in unique_objects:
                unique_objects.append(sample.obj_b)
        return unique_objects


class GroupSimilarityDataLoader(SimilarityDataLoader[SimilarityGroupSample]):
    @classmethod
    def collate_fn(cls, batch: List[SimilarityGroupSample]):
        features = [record.obj for record in batch]
        labels = {"groups": torch.LongTensor([record.group for record in batch])}
        return features, labels

    @classmethod
    def fetch_unique_objects(
        cls, batch: List[SimilarityGroupSample]
    ) -> List[SimilarityGroupSample]:
        unique_objects = []
        for sample in batch:
            if sample.obj not in unique_objects:
                unique_objects.append(sample.obj)
        return unique_objects
