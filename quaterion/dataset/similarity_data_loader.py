from typing import Any, List, Generic, Tuple, Dict

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data.dataloader import T_co

from quaterion.dataset.indexing_dataset import IndexingIterableDataset, IndexingDataset
from quaterion.dataset.similarity_samples import (
    SimilarityPairSample,
    SimilarityGroupSample,
)


class SimilarityDataLoader(DataLoader, Generic[T_co]):
    """SimilarityDataLoader is a special version of :class:`~torch.utils.data.DataLoader`
    which works with similarity samples.

    SimilarityDataLoader will automatically assign dummy collate_fn for debug purposes,
    it will be overwritten once dataloader is used for training.

    Required collate function should be defined individually for each encoder
    by overwriting :meth:`~quaterion_models.encoders.encoder.Encoder.get_collate_fn`

    Args:
        dataset: Dataset which outputs similarity samples
        **kwargs: Parameters passed directly into :meth:`~torch.utils.data.DataLoader.__init__`
    """

    def __init__(self, dataset: Dataset, **kwargs):

        if "collate_fn" not in kwargs:
            kwargs["collate_fn"] = self.__class__.pre_collate_fn
        self._original_dataset = dataset
        self._original_params = kwargs
        super().__init__(self._wrap_dataset(dataset), **kwargs)

    @classmethod
    def _wrap_dataset(cls, dataset: Dataset) -> Dataset:
        if isinstance(dataset, IterableDataset):
            return IndexingIterableDataset(dataset)
        else:
            return IndexingDataset(dataset)

    @property
    def original_params(self) -> Dict[str, Any]:
        """Initialization params of the original dataset."""
        return self._original_params

    @classmethod
    def pre_collate_fn(cls, batch: List[T_co]):
        """Function applied to batch before actual collate.

        Splits batch into features - arguments of prediction and labels - targets.
        Encoder-specific `collate_fn` will then be applied to feature list only.
        Loss functions consumes labels from this function without any additional transformations.

        Args:
            batch: List of similarity samples

        Returns:
            - ids of the features
            - features batch
            - labels batch
        """
        sample_ids, similarity_samples = list(zip(*batch))
        sample_ids = list(sample_ids)
        labels = cls.collate_labels(similarity_samples)
        features, feature_ids = cls.flatten_objects(
            batch=similarity_samples, hash_ids=sample_ids
        )
        return feature_ids, features, labels

    @classmethod
    def collate_labels(cls, batch: List[T_co]) -> Dict[str, torch.Tensor]:
        """Collate function for labels

        Convert labels into tensors, suitable for loss passing directly into loss functions and
        metric estimators.

        Args:
            batch: List of similarity samples

        Returns:
            Collated labels
        """
        raise NotImplementedError()

    @classmethod
    def flatten_objects(
        cls, batch: List[T_co], hash_ids: List[int]
    ) -> Tuple[List[Any], List[int]]:
        """Retrieve and enumerate objects from similarity samples.

        Each individual object should be used as input for the encoder.
        Additionally, associates hash_id with each feature, if there are more than one feature in
        the sample - generates new unique ids based on input one.

        Args:
            batch: List of similarity samples
            hash_ids: pseudo-random ids of the similarity samples

        Returns:
            - List of input features for encoder collate
            - List of ids, associated with each feature
        """
        raise NotImplementedError()


class PairsSimilarityDataLoader(SimilarityDataLoader[SimilarityPairSample]):
    def __init__(self, dataset: Dataset[SimilarityPairSample], **kwargs):
        super().__init__(dataset, **kwargs)

    @classmethod
    def collate_labels(
        cls, batch: List[SimilarityPairSample]
    ) -> Dict[str, torch.Tensor]:
        """Collate function for labels of :class:`~quaterion.dataset.similarity_samples.SimilarityPairSample`

        Convert labels into tensors, suitable for loss passing directly into loss functions and
        metric estimators.

        Args:
            batch: List of :class:`~quaterion.dataset.similarity_samples.SimilarityPairSample`

        Returns:
            Collated labels:
            - labels - tensor of scores for each input pair
            - pairs - pairs of id offsets of features, associated with respect labels
            - subgroups - subgroup id for each featire

        Examples:

            >>> labels_batch = PairsSimilarityDataLoader.collate_labels(
            ...     [
            ...         SimilarityPairSample(
            ...             obj_a="1st_pair_1st_obj", obj_b="1st_pair_2nd_obj", score=1.0, subgroup=0
            ...         ),
            ...         SimilarityPairSample(
            ...             obj_a="2nd_pair_1st_obj", obj_b="2nd_pair_2nd_obj", score=0.0, subgroup=1
            ...         ),
            ...     ]
            ... )
            >>> labels_batch['labels']
            tensor([1., 0.])
            >>> labels_batch['pairs']
            tensor([[0, 2],
                    [1, 3]])
            >>> labels_batch['subgroups']
            tensor([0., 1., 0., 1.])

        """
        labels = {
            "pairs": torch.LongTensor([[i, i + len(batch)] for i in range(len(batch))]),
            "labels": torch.Tensor([record.score for record in batch]),
            "subgroups": torch.Tensor([record.subgroup for record in batch] * 2),
        }
        return labels

    @classmethod
    def flatten_objects(
        cls, batch: List[SimilarityPairSample], hash_ids: List[int]
    ) -> Tuple[List[Any], List[int]]:
        res_features = []
        res_ids = []

        # Preserve same order as in `collate_labels`
        for hash_id, item in zip(hash_ids, batch):
            res_features.append(item.obj_a)
            res_ids.append(hash_id)

        for hash_id, item in zip(hash_ids, batch):
            res_features.append(item.obj_b)
            res_ids.append(hash_id + 1)  # Hashes are pseudo-random, so it is ok

        return res_features, res_ids


class GroupSimilarityDataLoader(SimilarityDataLoader[SimilarityGroupSample]):
    def __init__(self, dataset: Dataset[SimilarityGroupSample], **kwargs):
        super().__init__(dataset, **kwargs)

    @classmethod
    def collate_labels(
        cls, batch: List[SimilarityGroupSample]
    ) -> Dict[str, torch.Tensor]:
        """Collate function for labels

        Convert labels into tensors, suitable for loss passing directly into loss functions and
        metric estimators.

        Args:
            batch: List of :class:`~quaterion.dataset.similarity_samples.SimilarityGroupSample`

        Returns:
            Collated labels:
            - groups -- id of the group for each feature object

        Examples:

            >>> GroupSimilarityDataLoader.collate_labels(
            ...     [
            ...         SimilarityGroupSample(obj="orange", group=0),
            ...         SimilarityGroupSample(obj="lemon", group=0),
            ...         SimilarityGroupSample(obj="apple", group=1)
            ...     ]
            ... )
            {'groups': tensor([0, 0, 1])}
        """
        labels = {"groups": torch.LongTensor([record.group for record in batch])}
        return labels

    @classmethod
    def flatten_objects(
        cls, batch: List[SimilarityGroupSample], hash_ids: List[int]
    ) -> Tuple[List[Any], List[int]]:
        return [sample.obj for sample in batch], hash_ids
