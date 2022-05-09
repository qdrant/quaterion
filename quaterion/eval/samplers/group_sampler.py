import random
from typing import Tuple

import torch

from quaterion_models import MetricModel

from quaterion.eval.accumulators import GroupAccumulator
from quaterion.eval.group import GroupMetric
from quaterion.eval.samplers import BaseSampler
from torch.utils.data import Dataset, DataLoader
from quaterion.dataset.similarity_data_loader import GroupSimilarityDataLoader


class GroupSampler(BaseSampler):
    """Perform selection of embeddings and targets for group based tasks."""

    def __init__(self, sample_size=-1, encode_batch_size=16):
        super().__init__(sample_size)
        self.encode_batch_size = encode_batch_size
        self.accumulator = GroupAccumulator()

    def accumulate(self, dataset: Dataset, model):
        dataloader = DataLoader(dataset, batch_size=self.encode_batch_size)
        collate_labels = GroupSimilarityDataLoader.collate_labels

        for batch in dataloader:
            objects = [sample.object for sample in batch]
            self.accumulator.update(model.encode(objects), **collate_labels(batch))
        self.accumulator.set_filled()

    def sample(
        self, dataset: Dataset, metric: GroupMetric, model: MetricModel
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample embeddings and targets for groups based tasks.

        Args:
            dataset: ...,
            metric: GroupMetric instance with accumulated embeddings and groups
            model: ...,

        Returns:
            torch.Tensor, torch.Tensor: metrics labels and computed distance matrix
        """
        if not self.accumulator.filled:
            self.accumulate(dataset, model)

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

    def reset(self):
        self.accumulator.reset()
