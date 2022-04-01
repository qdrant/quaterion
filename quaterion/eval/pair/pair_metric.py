import torch
from torch import Tensor, LongTensor

from quaterion.distances import Distance
from quaterion.eval.base_metric import BaseMetric


class PairMetric(BaseMetric):
    """Base class for metrics computation for pair based data

    Args:
        distance_metric_name: name of a distance metric to calculate distance or similarity
            matrices. Available names could be found in :class:`~quaterion.distances.Distance`.

    Provides default implementations for embeddings and labels accumulation, distance and
    interaction matrices calculation.
    """

    def __init__(self, distance_metric_name: Distance = Distance.COSINE):
        super().__init__(distance_metric_name)
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
