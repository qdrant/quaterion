import torch
from torch import Tensor

from quaterion.distances import Distance
from quaterion.eval.base_metric import BaseMetric


class GroupMetric(BaseMetric):
    """Base class for group metrics

    Provide default implementation for embeddings and groups accumulation.

    Args:
        distance_metric_name: name of a distance metric to calculate distance or similarity
            matrices. Available names could be found in :class:`~quaterion.distances.Distance`.
    """

    def __init__(self, distance_metric_name: Distance = Distance.COSINE):
        super().__init__(distance_metric_name)
        self._groups = []

    @property
    def groups(self):
        return torch.cat(self._groups)

    def update(self, embeddings: Tensor, groups: torch.LongTensor, device=None) -> None:
        """Process and accumulate batch

        Args:
            embeddings: embeddings to accumulate
            groups: groups to distinguish similar and dissimilar objects.
            device: device to store calculated embeddings and groups on.
        """
        embeddings = embeddings.detach()
        groups = groups.detach()
        if device:
            embeddings = embeddings.to(device)
            groups = groups.to(device)

        self._embeddings.append(embeddings)
        self._groups.append(groups)

    def compute(self) -> Tensor:
        raise NotImplementedError()

    def reset(self):
        """Reset accumulated embeddings and groups"""
        super().reset()
        self._groups = []
