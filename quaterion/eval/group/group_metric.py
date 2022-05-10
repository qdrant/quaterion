import torch
from torch import Tensor

from quaterion.distances import Distance
from quaterion.eval.base_metric import BaseMetric
from quaterion.eval.accumulators import GroupAccumulator


class GroupMetric(BaseMetric):
    """Base class for group metrics

    Args:
        distance_metric_name: name of a distance metric to calculate distance or similarity
            matrices. Available names could be found in :class:`~quaterion.distances.Distance`.

    Provides default implementations for distance and interaction matrices calculation.
    Accumulates embeddings and groups in an accumulator.
    """

    def __init__(
        self,
        distance_metric_name: Distance = Distance.COSINE,
    ):
        super().__init__(
            distance_metric_name=distance_metric_name,
        )
        self.accumulator = GroupAccumulator()

    def update(self, embeddings: Tensor, groups: torch.LongTensor, device=None) -> None:
        """Process and accumulate batch

        Args:
            embeddings: embeddings to accumulate
            groups: groups to distinguish similar and dissimilar objects.
            device: device to store calculated embeddings and groups on.
        """
        self.accumulator.update(embeddings, groups, device)

    def reset(self):
        """Reset accumulated embeddings, groups"""
        self.accumulator.reset()

    @staticmethod
    def prepare_labels(groups: Tensor):
        """Compute metric labels based on samples groups

        Args:
            groups: groups to distinguish similar and dissimilar objects

        Returns:
            target: torch.Tensor -  labels to be used during metric computation
        """
        group_matrix = groups.repeat(groups.shape[0], 1)
        # objects with the same groups are true, others are false

        group_mask = (group_matrix == groups.unsqueeze(1)).bool()
        # exclude obj
        group_mask[torch.eye(group_mask.shape[0], dtype=torch.bool)] = False
        return group_mask

    def compute(self, embeddings: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
        """Compute metric value

        Args:
            embeddings: embeddings to calculate metrics on
            groups: groups to calculate labels
        Returns:
            torch.Tensor - computed metric
        """
        labels, distance_matrix = self.precompute(embeddings, groups=groups)
        return self.raw_compute(distance_matrix, labels)

    def evaluate(self) -> torch.Tensor:
        """Perform metric computation with accumulated state"""
        return self.compute(**self.accumulator.state)

    def raw_compute(
        self, distance_matrix: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Perform metric computation on ready distance_matrix and labels

        This method does not make any data and labels preparation.
        It is assumed that `distance_matrix` has already been calculated, required changes such
        masking distance from an element to itself have already been applied and corresponding
        `labels` have been prepared.

        Args:
            distance_matrix: distance matrix ready to metric computation
            labels:  labels ready to metric computation with the same shape as `distance_matrix`.
                Possible values are in {0, 1}.

        Returns:
            torch.Tensor - calculated metric value
        """
        raise NotImplementedError()
