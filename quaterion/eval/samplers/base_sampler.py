from typing import Tuple

import torch

from quaterion.eval.base_metric import BaseMetric


class BaseSampler:
    """Sample part of embeddings and targets to perform metric calculation on a part of the data

    Sampler allows reducing amount of time and resources to calculate a distance matrix.
    Instead of calculation of squared matrix with shape (num_embeddings, num_embeddings), it
    selects embeddings and computes matrix with shape (sample_size, num_embeddings).

        Args:
            sample_size: amount of objects to select.

    """
    def __init__(self, sample_size=-1):
        self.sample_size = sample_size

    def sample(self, metric: BaseMetric) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select embeddings and targets

        Args:
            metric: metric instance with accumulated embeddings, targets and with a method
                to compute final labels based on targets.

        Returns:
            labels, distance_matrix: metrics labels and computed distance matrix
        """
        raise NotImplementedError()
