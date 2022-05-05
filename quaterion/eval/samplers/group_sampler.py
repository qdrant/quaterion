import random
from typing import Tuple

import torch

from quaterion.eval.group import GroupMetric
from quaterion.eval.samplers import BaseSampler


class GroupSampler(BaseSampler):
    """Perform selection of embeddings and targets for group based tasks."""

    def __init__(self, sample_size=-1):
        super().__init__(sample_size)

    def sample(self, metric: GroupMetric) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample embeddings and targets for groups based tasks.

        Args:
            metric: GroupMetric instance with accumulated embeddings and groups

        Returns:
            torch.Tensor, torch.Tensor: metrics labels and computed distance matrix
        """
        labels = metric.compute_labels(metric.groups)

        max_sample_size = metric.embeddings.shape[0]

        if self.sample_size > 0:
            sample_size = min(self.sample_size, max_sample_size)
        else:
            sample_size = max_sample_size

        sample_indices = torch.LongTensor(
            random.sample(range(max_sample_size), k=sample_size)
        )
        labels = labels[sample_indices]
        distance_matrix = metric.distance.distance_matrix(
            metric.embeddings[sample_indices], metric.embeddings
        )
        return labels.float(), distance_matrix
