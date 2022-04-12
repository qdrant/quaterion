import torch

from quaterion.distances import Distance
from quaterion.eval.pair import PairMetric


class RetrievalReciprocalRank(PairMetric):
    """Calculates retrieval reciprocal rank for pair based datasets

    Calculates the reciprocal of the rank at which the first relevant document was retrieved.

    Args:
        distance_metric_name: name of a distance metric to calculate distance or similarity
            matrices. Available names could be found in :class:`~quaterion.distances.Distance`.

    Example:

        Response on a query returned 10 documents, 3 of them are relevant. Assume positions of
        relevant documents are [2, 5, 9]. Then retrieval reciprocal rank being calculated as
        1/2 = 0.5.

    """

    def __init__(self, distance_metric_name: Distance = Distance.COSINE):
        super().__init__(distance_metric_name)

    def compute(self):
        """Calculates retrieval reciprocal rank"""
        distance_matrix, labels = self.precompute()
        distance_matrix[torch.eye(distance_matrix.shape[0], dtype=torch.bool)] = (
            torch.max(distance_matrix) + 1
        )
        return retrieval_reciprocal_rank(distance_matrix, labels)


def retrieval_reciprocal_rank(distance_matrix, labels):
    """Calculates retrieval precision@k given distance matrix, labels and k

    Args:
        distance_matrix: distance matrix having max possible distance value on a diagonal
        labels: labels matrix having False or 0. on a diagonal

    Returns:
        torch.Tensor: retrieval precision@k for each row in tensor
    """
    indices = torch.argsort(distance_matrix, dim=1)
    target = labels.gather(1, indices)
    position = torch.nonzero(target)
    metric = 1.0 / (position[:, 1] + 1.0)
    return metric
