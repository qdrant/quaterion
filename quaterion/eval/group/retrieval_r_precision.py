from typing import Optional, Callable

import torch
from torch import Tensor

from quaterion.distances import Distance
from quaterion.eval.group import GroupMetric


class RetrievalRPrecision(GroupMetric):
    """Class to compute the retrieval R-precision score for group based data

    Retrieval R-Precision is the ratio of `r/R`, where `R` is the number of the relevant documents
    for a given query in the collection, and `r` is the number of the truly relevant documents
    found in the `R` highest scored results for that query.

    Args:
        distance_metric_name: name of a distance metric to calculate distance or similarity
            matrices. Available names could be found in :class:`~quaterion.distances.Distance`.

    Example:

        Suppose that a collection contains 20 relevant documents for our query, and the model can
        retrieve 15 of them in the 20 highest scored results, then Retrieval R-Precision is
        calculated as r/R = 15/20 = 0.75.

    """

    def __init__(
        self,
        compute_on_step=True,
        distance_metric_name: Distance = Distance.COSINE,
        reduce_func: Optional[Callable] = None,
    ):
        if reduce_func is not None:
            raise ValueError(
                "RetrievalRPrecision does not support custom `reduce_func`"
            )

        super().__init__(
            compute_on_step=compute_on_step,
            distance=distance_metric_name,
            reduce_func=reduce_func,
        )

    def _compute(self, embeddings, *, sample_indices=None, **target):
        """Calculates retrieval R-precision

        Returns:
            Tensor: zero-size tensor
        """
        labels, distance_matrix = self.precompute(
            embeddings, target["groups"], sample_indices=sample_indices
        )
        return retrieval_r_precision(distance_matrix, labels)


def retrieval_r_precision(distance_matrix: torch.Tensor, labels: torch.Tensor):
    """Calculates retrieval r precision given distance matrix and labels

    Args:
        distance_matrix: distance matrix having max possible distance value on a diagonal
        labels: labels matrix having False or 0. on a diagonal

    Returns:
        torch.Tensor: mean retrieval r precision
    """
    # number of members for group which is on i-th position in groups
    relevant_numbers = labels.sum(dim=-1)
    nearest_to_furthest_ind = torch.argsort(distance_matrix, dim=-1, descending=False)
    sorted_by_distance = torch.gather(labels, dim=-1, index=nearest_to_furthest_ind)
    top_k_mask = (
        torch.arange(0, labels.shape[0], step=1).repeat(labels.shape[0], 1)
        < relevant_numbers
    )
    metric = sorted_by_distance[top_k_mask].float()
    return metric.mean()
