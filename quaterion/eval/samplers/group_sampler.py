import random
from typing import Tuple, Sized

import torch

from quaterion_models import MetricModel

from quaterion.eval.accumulators import GroupAccumulator
from quaterion.eval.group import GroupMetric
from quaterion.eval.samplers import BaseSampler
from quaterion.dataset.similarity_data_loader import GroupSimilarityDataLoader


class GroupSampler(BaseSampler):
    """Perform selection of embeddings and targets for group based tasks."""

    def __init__(self, sample_size=-1, encode_batch_size=16):
        super().__init__(sample_size)
        self.encode_batch_size = encode_batch_size
        self.accumulator = GroupAccumulator()

    def accumulate(self, model: MetricModel, dataset: Sized):
        """Encodes objects and accumulates embeddings with the corresponding raw labels

        Args:
            model: model to encode objects
            dataset: Sized object, like list, tuple, torch.utils.data.Dataset, etc. to accumulate
        """

        dataset_size = len(dataset)
        step = min(dataset_size, self.encode_batch_size)
        for slice_start_index in range(0, dataset_size, step):
            slice_end_index = slice_start_index + step
            slice_end_index = (
                slice_end_index if slice_end_index < dataset_size else dataset_size
            )
            input_batch = [
                dataset[index] for index in range(slice_start_index, slice_end_index)
            ]
            batch_labels = GroupSimilarityDataLoader.collate_labels(input_batch)

            features = [similarity_sample.obj for similarity_sample in input_batch]

            embeddings = model.encode(
                features, batch_size=self.encode_batch_size, to_numpy=False
            )
            self.accumulator.update(embeddings, **batch_labels)

        self.accumulator.set_filled()

    def reset(self):
        """Reset accumulated state"""
        self.accumulator.reset()

    def sample(
        self, dataset: Sized, metric: GroupMetric, model: MetricModel
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample embeddings and targets for groups based tasks.

        Args:
            dataset: Sized object, like list, tuple, torch.utils.data.Dataset, etc. to sample
            metric: GroupMetric instance to compute final labels representation
            model: model to encode objects

        Returns:
            torch.Tensor, torch.Tensor: metrics labels and computed distance matrix
        """
        if not self.accumulator.filled:
            self.accumulate(model, dataset)

        embeddings = self.accumulator.embeddings
        labels = metric.compute_labels(self.accumulator.groups)

        max_sample_size = embeddings.shape[0]

        if self.sample_size > 0:
            sample_size = min(self.sample_size, max_sample_size)
        else:
            sample_size = max_sample_size

        sample_indices = torch.LongTensor(
            random.sample(range(max_sample_size), k=sample_size)
        )
        labels = labels[sample_indices]
        distance_matrix = metric.distance.distance_matrix(
            embeddings[sample_indices], embeddings
        )
        return labels.float(), distance_matrix
