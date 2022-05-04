from typing import Optional, Callable

import torch

from quaterion.eval.pair import PairMetric
from quaterion.distances import Distance


class RetrievalPrecision(PairMetric):
    """Calculates retrieval precision@k for pair based datasets

    Args:
        k: number of documents among which to search a relevant one
        distance_metric_name: name of a distance metric to calculate distance or similarity
            matrices. Available names could be found in :class:`~quaterion.distances.Distance`.
        reduce_func: function to reduce calculated metric. E.g. `torch.mean`, `torch.max` and
            others. `functools.partial` might be useful if you want to capture some custom
            arguments.

    Example:

        Assume `k` is 4. Then only 4 documents being retrieved as a query response. Only 2 of them
        are relevant and score will be 2/4 = 0.5.

    Note:
        If `k` is greater than overall amount of relevant documents, then precision@k will always
        have score < 1.
    """

    def __init__(
        self,
        k=1,
        distance_metric_name: Distance = Distance.COSINE,
        reduce_func: Optional[Callable] = torch.mean,
    ):
        super().__init__(
            distance_metric_name=distance_metric_name,
        )
        self.k = k
        self.reduce_func = reduce_func
        if self.k < 1:
            raise ValueError("k must be greater than 0")

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
        value = retrieval_precision(distance_matrix, labels, self.k)
        if self.reduce_func is not None:
            value = self.reduce_func(value)
        return value


def retrieval_precision(distance_matrix: torch.Tensor, labels: torch.Tensor, k: int):
    """Calculates retrieval precision@k given distance matrix, labels and k

    Args:
        distance_matrix: distance matrix having max possible distance value on a diagonal
        labels: labels matrix having False or 0. on a diagonal
        k: number of documents to retrieve

    Returns:
        torch.Tensor: retrieval precision@k for each row in tensor
    """
    metric = (
        labels.gather(1, distance_matrix.topk(k, dim=-1, largest=False)[1])
        .sum(dim=1)
        .float()
    ) / k
    return metric
