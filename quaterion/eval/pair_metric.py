import torch
from torch import Tensor, LongTensor

from quaterion.eval.base_metric import BaseMetric


class PairMetric(BaseMetric):
    """Base class for metrics computation for pair based data

    Args:
        distance_metric: function for distance matrix computation. Possible choice might be one of
            :class:`~quaterion.loss.metrics.SiameseDistanceMetric` methods.

    Provides default implementations for embeddings and labels accumulation, distance and
    interaction matrices calculation.
    """

    def __init__(self, distance_metric):
        super().__init__(distance_metric)
        self.pairs = LongTensor()
        self.labels = Tensor()
        self.subgroups = Tensor()

    def compute(self) -> Tensor:
        raise NotImplementedError()

    def precompute(self):
        """Perform distance matrix calculation and create an interaction matrix based on labels."""
        distance_matrix = self.calculate_distances()
        target = torch.zeros_like(distance_matrix)
        # todo: subgroups should also be taken into account
        target[self.pairs[:, 0], self.pairs[:, 1]] = self.labels
        target[self.pairs[:, 1], self.pairs[:, 0]] = self.labels
        return distance_matrix, target

    def update(
        self,
        embeddings: Tensor,
        pairs: LongTensor,
        labels: Tensor,
        subgroups: Tensor,
        device="cpu",
    ) -> None:
        """Process and accumulate embeddings and corresponding labels

        Args:
            embeddings: embeddings to accumulate
            pairs: indices to match objects from the same pair
            labels: labels to determine whether objects in pair is similar or not
            subgroups: subgroups to find related objects among different pairs
            device: device to store calculated embeddings and labels on.
        """
        self.embeddings = torch.cat([self.embeddings, embeddings.detach().to(device)])
        self.pairs = torch.cat([self.pairs, pairs.to(device)])
        self.labels = torch.cat([self.labels, labels])
        self.subgroups = torch.cat([self.subgroups, subgroups.to(device)])


class RetrievalReciprocalRank(PairMetric):
    """Calculates retrieval reciprocal rank for pair based datasets

    Calculates the reciprocal of the rank at which the first relevant document was retrieved.

    Args:
        distance_metric: function for distance matrix computation. Possible choice might be one of
            :class:`~quaterion.loss.metrics.SiameseDistanceMetric` methods.

    Examples:

        Response on a query returned 10 documents, 3 of them are relevant. Assume positions of
        relevant documents are [2, 5, 9]. Then retrieval reciprocal rank being calculated as
        1/2 = 0.5.

    """

    def __init__(self, distance_metric):
        super().__init__(distance_metric)

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

    def __init__(self, distance_metric, k=1):
        super().__init__(distance_metric)
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
