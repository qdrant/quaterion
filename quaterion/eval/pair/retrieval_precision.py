from typing import Optional, Callable

import torch

from quaterion.eval.pair import PairMetric
from quaterion.distances import Distance


class RetrievalPrecision(PairMetric):
    """Calculates retrieval precision@k for pair based datasets

    Args:
        distance_metric_name: name of a distance metric to calculate distance or similarity
            matrices. Available names could be found in :class:`~quaterion.distances.Distance`.
        k: number of documents among which to search a relevant one

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
        compute_on_step=True,
        distance: Distance = Distance.COSINE,
        reduce_func: Optional[Callable] = torch.mean,
    ):
        super().__init__(
            compute_on_step=compute_on_step, distance=distance, reduce_func=reduce_func
        )
        self.k = k
        if self.k < 1:
            raise ValueError("k must be greater than 0")

    def _compute(self, embeddings, sample_indices=None, **targets):
        labels, distance_matrix = self.precompute(
            embeddings,
            targets["labels"],
            targets["pairs"],
            targets["subgroups"],
            sample_indices,
        )
        return retrieval_precision(distance_matrix, labels, self.k)


def retrieval_precision(distance_matrix, labels, k):
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
