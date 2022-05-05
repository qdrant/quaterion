import random
from typing import Tuple

import torch

from quaterion.eval.pair import PairMetric
from quaterion.eval.samplers import BaseSampler


class PairSampler(BaseSampler):
    """Perform selection of embeddings and targets for pairs based tasks.

    Args:
        distinguish: bool - determines whether to compare all objects each-to-each, or to
            compare only `obj_a` to `obj_b`. If true - compare only `obj_a` to `obj_b`. Reduces
            matrix size quadratically.

    """

    def __init__(self, sample_size=-1, distinguish=False):
        super().__init__(sample_size)
        self.distinguish = distinguish

    def sample(self, metric: PairMetric) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample embeddings and targets for pairs based tasks.

        Args:
            metric: PairMetric instance with accumulated embeddings, labels, pairs and subgroups

        Returns:
            torch.Tensor, torch.Tensor: metrics labels and computed distance matrix
        """
        embeddings = metric.embeddings
        pairs = metric.pairs

        labels = metric.compute_labels(metric.labels, pairs, metric.subgroups)

        embeddings_num = embeddings.shape[0]
        max_sample_size = embeddings_num if not self.distinguish else pairs.shape[0]

        if self.sample_size > 0:
            sample_size = min(self.sample_size, max_sample_size)
        else:
            sample_size = max_sample_size

        sample_indices = torch.LongTensor(
            random.sample(range(max_sample_size), k=sample_size)
        )

        labels = labels[sample_indices]

        if self.distinguish:
            ref_embeddings = embeddings[pairs[sample_indices][:, 0]]
            embeddings = embeddings[pairs[:, 1]]
            labels = labels[:, pairs[:, 1]]
        else:
            ref_embeddings = embeddings[sample_indices]

        distance_matrix = metric.distance.distance_matrix(ref_embeddings, embeddings)
        return labels.float(), distance_matrix
