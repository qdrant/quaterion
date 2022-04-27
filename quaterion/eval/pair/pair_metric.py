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
        self._pairs = []
        self._labels = []
        self._subgroups = []

    @property
    def pairs(self):
        return self._pairs

    @property
    def labels(self):
        return self._labels

    @property
    def subgroups(self):
        return self._subgroups

    def compute(self) -> Tensor:
        raise NotImplementedError()

    def precompute(self):
        """Perform distance matrix calculation and create an interaction matrix based on labels."""
        distance_matrix = self.calculate_distances()
        target = torch.zeros_like(distance_matrix)
        pairs = self.pairs
        labels = self.labels
        # todo: subgroups should also be taken into account
        target[pairs[:, 0], pairs[:, 1]] = labels
        target[pairs[:, 1], pairs[:, 0]] = labels
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
        embeddings = embeddings.detach()
        pairs = pairs.detach()
        labels = labels.detach()
        subgroups = subgroups.detach()
        if device:
            embeddings = embeddings.to(device)
            pairs = pairs.to(device)
            labels = labels.to(device)
            subgroups = subgroups.to(device)
        self._embeddings.append(embeddings)
        self._pairs.append(pairs)
        self._labels.append(labels)
        self._subgroups.append(subgroups)

    def reset(self):
        """Reset accumulated embeddings and labels"""
        super().reset()
        self._pairs = []
        self._labels = []
        self._subgroups = []
