from typing import Any, List, Generic, Tuple, Dict

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import T_co

from quaterion.dataset.similarity_samples import (
    SimilarityPairSample,
    SimilarityGroupSample,
)


class SimilarityDataLoader(DataLoader, Generic[T_co]):
    @classmethod
    def pre_collate_fn(
        cls, batch: List[T_co]
    ) -> Tuple[List[Any], Dict[str, torch.Tensor]]:
        """
        Function applied to batch before actual collate.
        Splits bach into features - arguments of prediction and labels - targets.
        Encoder-specific `collate_fn`_s will then be applied to feature list only.
        Loss functions consumes labels from this function without any additional transformations.

        Args:
            batch: List of records, combined into batch directly from Dataset

        Returns:
            Lists of Features and List of labels
        """
        raise NotImplementedError()

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
    def __init__(self, dataset: Dataset[SimilarityPairSample], **kwargs):
        """
        PairsSimilarityDataLoader is a special version of :class:`~torch.utils.data.DataLoader`
        which works with similarity groups.

        PairsSimilarityDataLoader will automatically assign dummy collate_fn for debug purposes,
        it will be overwritten once dataloader is used for training.

        Required collate function should be defined individually for each encoder
        by overwriting :meth:`~quaterion_models.encoders.encoder.Encoder.get_collate_fn`

        Args:
            dataset: Dataset which outputs :class:`~quaterion.dataset.similarity_samples.SimilarityGroupSample`
            **kwargs: Parameters passed directly into :meth:`~torch.utils.data.DataLoader.__init__`
        """
        if "collate_fn" not in kwargs:
            kwargs["collate_fn"] = self.pre_collate_fn
        super().__init__(dataset, **kwargs)

    @classmethod
    def pre_collate_fn(
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
    def __init__(self, dataset: Dataset[SimilarityPairSample], **kwargs):
        """
        GroupSimilarityDataLoader is a special version of :class:`~torch.utils.data.DataLoader`
        which works with similarity groups.

        GroupSimilarityDataLoader will automatically assign dummy collate_fn for debug purposes,
        it will be overwritten once dataloader is used for training.

        Required collate function should be defined individually for each encoder
        by overwriting :meth:`~quaterion_models.encoders.encoder.Encoder.get_collate_fn`

        Args:
            dataset: Dataset which outputs :class:`~quaterion.dataset.similarity_samples.SimilarityGroupSample`
            **kwargs: Parameters passed directly into :meth:`~torch.utils.data.DataLoader.__init__`
        """
        if "collate_fn" not in kwargs:
            kwargs["collate_fn"] = self.pre_collate_fn
        super().__init__(dataset, **kwargs)

    @classmethod
    def pre_collate_fn(
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
            GroupSimilarityDataLoader.pre_collate_fn(
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
