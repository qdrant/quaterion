from typing import Optional, Callable

import torch

from quaterion.distances import Distance
from quaterion.eval.pair import PairMetric


class RetrievalReciprocalRank(PairMetric):
    """Calculates retrieval reciprocal rank for pair based datasets

    Calculates the reciprocal of the rank at which the first relevant document was retrieved.

    Args:
        distance_metric_name: name of a distance metric to calculate distance or similarity
            matrices. Available names could be found in :class:`~quaterion.distances.Distance`.
        reduce_func: function to reduce calculated metric. E.g. `torch.mean`, `torch.max` and
            others. `functools.partial` might be useful if you want to capture some custom arguments.

    Example:

        Response on a query returned 10 documents, 3 of them are relevant. Assume positions of
        relevant documents are [2, 5, 9]. Then retrieval reciprocal rank being calculated as
        1/2 = 0.5.

    """

    def __init__(
        self,
        distance_metric_name: Distance = Distance.COSINE,
        reduce_func: Optional[Callable] = torch.mean,
    ):
        self.reduce_func = reduce_func
        super().__init__(
            distance_metric_name=distance_metric_name,
        )

    def raw_compute(self, distance_matrix: torch.Tensor, labels: torch.Tensor):
        """Compute retrieval precision

        Args:
            distance_matrix: matrix with distances between embeddings. Assumed that distance from
                embedding to itself is meaningless. (e.g. equal to max element of matrix + 1)
            labels: labels to compute metric. Assumed that label from object to itself has been
                made meaningless. (E.g. was set to 0)

        Returns:
            torch.Tensor - computed metric
        """
        value = retrieval_reciprocal_rank(distance_matrix, labels)
        if self.reduce_func is not None:
            value = self.reduce_func(value)
        return value


def retrieval_reciprocal_rank(distance_matrix: torch.Tensor, labels: torch.Tensor):
    """Calculates retrieval reciprocal rank given distance matrix and labels

    Args:
        distance_matrix: distance matrix having max possible distance value on a diagonal
        labels: labels matrix having False or 0. on a diagonal

    Returns:
        torch.Tensor: retrieval reciprocal rank
    """
    indices = torch.argsort(distance_matrix, dim=1)
    target = labels.gather(1, indices)
    position = torch.nonzero(target)
    metric = 1.0 / (position[:, 1] + 1.0)
    return metric
