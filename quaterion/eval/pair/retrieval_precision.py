import torch

from quaterion.eval.pair import PairMetric
from quaterion.distances import Distance


class RetrievalPrecision(PairMetric):
    """Calculates retrieval precision@k for pair based datasets

    Args:
        distance_metric_name: name of a distance metric to calculate distance or similarity
            matrices. Available names could be found in :class:`~quaterion.distances.Distance`.
        k: number of documents among which to search a relevant one

    Examples:

        Assume `k` is 4. Then only 4 documents being retrieved as a query response. Only 2 of them
        are relevant and score will be 2/4 = 0.5.

    Note:
        If `k` is greater than overall amount of relevant documents, then precision@k will always
        have score < 1.
    """

    def __init__(self, distance_metric_name: Distance = Distance.COSINE, k=1):
        super().__init__(distance_metric_name)
        self.k = k
        if self.k < 1:
            raise ValueError("k must be greater than 0")

    def compute(self):
        """Calculates retrieval precision@k"""
        distance_matrix, target = self.precompute()
        # assign max dist to obj on diag to ignore distance from obj to itself
        distance_matrix[torch.eye(distance_matrix.shape[0], dtype=torch.bool)] = (
            torch.max(distance_matrix) + 1
        )
        metric = (
            target.gather(1, distance_matrix.topk(self.k, dim=1, largest=False)[1])
            .sum(dim=1)
            .float()
        ) / self.k
        return metric
