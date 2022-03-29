import torch

from quaterion.eval.base_metric import BaseMetric


class PairMetric(BaseMetric):
    """Base class for metrics computation for pair based data"""

    def compute(self):
        raise NotImplementedError()

    def precompute(self):
        """Perform distance matrix calculation and create an interaction matrix based on labels."""
        pairs = self.labels["pairs"]
        labels = self.labels["labels"]
        distance_matrix = self.calculate_distances()
        target = torch.zeros_like(distance_matrix)
        # todo: subgroups should also be taken into account
        target[pairs[:, 0], pairs[:, 1]] = labels
        target[pairs[:, 1], pairs[:, 0]] = labels
        return distance_matrix, target


class RetrievalReciprocalRank(PairMetric):
    """Calculates retrieval reciprocal rank for pair based datasets

    Calculates the reciprocal of the rank at which the first relevant document was retrieved.

    Args:
        encoder: :class:`~quaterion_models.encoders.encoder.Encoder` instance to calculate
            embeddings.
        distance_metric: function for distance matrix computation. Possible choice might be one of
            :class:`~quaterion.loss.metrics.SiameseDistanceMetric` methods.

    Examples:

        Response on a query returned 10 documents, 3 of them are relevant. Assume positions of
        relevant documents are [2, 5, 9]. Then retrieval reciprocal rank being calculated as
        1/2 = 0.5.

    """

    def __init__(self, encoder, distance_metric):
        super().__init__(encoder, distance_metric)

    def compute(self):
        """Calculates retrieval reciprocal rank"""
        distance_matrix, target = self.precompute()
        distance_matrix[torch.eye(distance_matrix.shape[0], dtype=torch.bool)] = (
            torch.max(distance_matrix) + 1
        )

        indices = torch.argsort(distance_matrix, dim=1)
        target = target.gather(1, indices)
        position = torch.nonzero(target)
        metric = 1.0 / (position[:, 1] + 1.0)
        return metric


class RetrievalPrecision(PairMetric):
    """Calculates retrieval precision@k for pair based datasets

    Args:
        encoder: :class:`~quaterion_models.encoders.encoder.Encoder` instance to calculate
            embeddings.
        distance_metric: function for distance matrix computation. Possible choice might be one of
            :class:`~quaterion.loss.metrics.SiameseDistanceMetric` methods.
        k: number of documents among which to search a relevant one

    Examples:

        Assume `k` is 4. Then only 4 documents being retrieved as a query response. Only 2 of them
        are relevant and score will be 2/4 = 0.5.

    Note:
        If `k` is greater than overall amount of relevant documents, then precision@k will always
        have score < 1.
    """

    def __init__(self, encoder, distance_metric, k=1):
        super().__init__(encoder, distance_metric)
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
