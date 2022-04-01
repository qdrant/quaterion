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
        self.groups = torch.LongTensor()

    def update(
        self, embeddings: Tensor, groups: torch.LongTensor, device="cpu"
    ) -> None:
        """Process and accumulate batch

        Args:
            embeddings: embeddings to accumulate
            groups: groups to distinguish similar and dissimilar objects.
            device: device to store calculated embeddings and groups on.
        """
        self.embeddings = torch.cat([self.embeddings, embeddings.detach().to(device)])
        self.groups = torch.cat([self.groups, groups.to(device)])

    def compute(self) -> Tensor:
        raise NotImplementedError()
