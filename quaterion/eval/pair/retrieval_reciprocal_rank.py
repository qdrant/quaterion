from typing import Optional, Callable

import torch

from quaterion.distances import Distance
from quaterion.eval.pair import PairMetric


class RetrievalReciprocalRank(PairMetric):
    """Calculates retrieval reciprocal rank for pair based datasets

    Calculates the reciprocal of the rank at which the first relevant document was retrieved.

    Args:
        compute_on_step: flag if metric should be calculated on each batch
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
        compute_on_step=True,
        distance_metric_name: Distance = Distance.COSINE,
        reduce_func: Optional[Callable] = torch.mean,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            distance_metric_name=distance_metric_name,
            reduce_func=reduce_func,
        )

    def _compute(
        self,
        embeddings: torch.Tensor,
        sample_indices: Optional[torch.Tensor] = None,
        **targets
    ):
        """Compute retrieval reciprocal precision

        Directly compute metric value.
        All additional logic: embeddings and targets preparations, using of cached result etc.
        should be done outside.

        Args:
            embeddings: embeddings to calculate metrics on
            sample_indices: indices of embeddings to sample if metric should be computed only on
                part of accumulated embeddings
            **targets: dict with labels, pairs and subgroups to compute final labels

        Returns:
            torch.Tensor - computed metric
        """
        labels, distance_matrix = self.precompute(
            embeddings,
            targets["labels"],
            targets["pairs"],
            targets["subgroups"],
            sample_indices,
        )
        return retrieval_reciprocal_rank(distance_matrix, labels)


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
