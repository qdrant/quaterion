from typing import Any, List, Generic, Dict, Tuple

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
        """Fetch unique objects to avoid calculation of repeated objects embeddings.

        Args:
            batch: batch of raw data

        Returns:
            List[Any]: list of unique objects

        """
        raise NotImplementedError()


class PairsSimilarityDataLoader(SimilarityDataLoader[SimilarityPairSample]):
    @classmethod
    def collate_fn(
        cls, batch: List[SimilarityPairSample]
    ) -> Tuple[List[Any], Dict[str, torch.Tensor]]:
        """Collate function for SimilarityPairSamples objects.

        Extract features from pairs and collects them in list.
        Construct labels dict with `pairs`, `labels` and `subgroups`.

        Args:
            batch: List of SimilarityPairSample objects

        Returns:
            Tuple[List[Any], Dict[str, torch.Tensor]]: tuple of features and labels

        Examples:
            ```
            PairsSimilarityDataLoader.collate_fn(
                [
                    SimilarityPairSample(
                        obj_a="1st_pair_1st_obj", obj_b="1st_pair_2nd_obj", score=1.0,
                    ),
                    SimilarityPairSample(
                        obj_a="2nd_pair_1st_obj",
                        obj_b="2nd_pair_2nd_obj",
                        score=0.0,
                        subgroup=1
                    ),
                ]
            )

            # result
            (
                # features
                [
                    '1st_pair_1st_obj',
                    '2nd_pair_1st_obj',
                    '1st_pair_2nd_obj',
                    '2nd_pair_2nd_obj'
                ],
                # labels
                {
                    'labels': tensor([1., 0.]),
                    'pairs': tensor([[0, 2], [1, 3]]),
                    'subgroups': tensor([0., 1., 0., 1.])
                }
            )
            ```
        """
        features = [record.obj_a for record in batch] + [record.obj_b for record in batch]
        labels = {
            "pairs": torch.LongTensor([[i, i + len(batch)] for i in range(len(batch))]),
            "labels": torch.Tensor([record.score for record in batch]),
            "subgroups": torch.Tensor([record.subgroup for record in batch] * 2),
        }
        return features, labels

    @classmethod
    def fetch_unique_objects(cls, batch: List[SimilarityPairSample]) -> List[Any]:
        """Fetch unique objects from SimilarityPairSample batch.

        Collect unique `obj_a` and `obj_b` from samples in a batch.

        Args:
            batch: List of SimilarityPairSample's

        Returns:
            List[Any]: list of unique `obj_a` and `obj_b` in batch
        """
        unique_objects = []
        for sample in batch:
            if sample.obj_a not in unique_objects:
                unique_objects.append(sample.obj_a)
            if sample.obj_b not in unique_objects:
                unique_objects.append(sample.obj_b)
        return unique_objects


class GroupSimilarityDataLoader(SimilarityDataLoader[SimilarityGroupSample]):
    @classmethod
    def collate_fn(
        cls, batch: List[SimilarityGroupSample]
    ) -> Tuple[List[Any], Dict[str, torch.Tensor]]:
        """Collate function for SimilarityGroupSamples objects.

        Extract features from pairs and collects them in list.
        Construct labels dict with `groups`.

        Args:
            batch: List of SimilarityGroupSample objects

        Returns:
            Tuple[List[Any], Dict[str, torch.Tensor]]: tuple of features and
                labels

        Examples:
            ```
            GroupSimilarityDataLoader.collate_fn(
                [
                    SimilarityGroupSample(
                        obj="orange",
                        group=0,
                    ),
                    SimilarityGroupSample(
                        obj="lemon",
                        group=0,
                    ),
                    SimilarityGroupSample(
                        obj="apple",
                        group=1,
                    )
                ]
            )

            # result
            (
                # features
                ['orange', 'lemon', 'apple'],
                # labels
                {'groups': tensor([0, 0, 1])}
            )
            ```
        """
        features = [record.obj for record in batch]
        labels = {"groups": torch.LongTensor([record.group for record in batch])}
        return features, labels

    @classmethod
    def fetch_unique_objects(cls, batch: List[SimilarityGroupSample]) -> List[Any]:
        """Fetch unique objects from SimilarityGroupSample batch.

        Collect unique `obj` from samples in a batch.

        Args:
            batch: List of SimilarityGroupSample's

        Returns:
            List[Any]: list of unique `obj` in batch
        """
        unique_objects = []
        for sample in batch:
            if sample.obj not in unique_objects:
                unique_objects.append(sample.obj)
        return unique_objects
