from typing import Tuple

import torch

from quaterion.distances import Distance


class BaseMetric:
    """Base class for evaluation metrics

    Provides a default implementation for distance matrix calculation.

    Args:
        distance_metric_name: name of a distance metric to calculate distance or similarity
            matrices. Available names could be found in :class:`~quaterion.distances.Distance`.
    """

    def __init__(
        self,
        distance_metric_name: Distance = Distance.COSINE,
    ):
        self.distance = Distance.get_by_name(distance_metric_name)
        self._distance_metric_name = distance_metric_name

    def compute(self, *args, **kwargs) -> torch.Tensor:
        """Compute metric value

        Args:
            args, kwargs - contain embeddings and targets required to compute metric.

        Returns:
            torch.Tensor - computed metric
        """
        raise NotImplementedError()

    def evaluate(self) -> torch.Tensor:
        """Perform metric computation with accumulated state"""
        raise NotImplementedError()

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
            labels: labels ready to metric computation with the same shape as `distance_matrix`.
                For `PairMetric` values are taken from `SimilarityPairSample.score`, for
                `GroupMetric` the possible values are in {0, 1}.

        Returns:
            torch.Tensor - calculated metric value
        """
        raise NotImplementedError()

    def precompute(
        self,
        embeddings: torch.Tensor,
        **targets,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepares data for computation

        Compute distance matrix and final labels based on groups.

        Args:
            embeddings: embeddings to compute metric value
            targets: objects to compute final labels

        Returns:
            torch.Tensor, torch.Tensor - labels and distance matrix
        """
        labels = self.prepare_labels(**targets)
        distance_matrix = self.distance.distance_matrix(embeddings).detach()
        self_mask = torch.eye(distance_matrix.shape[0], dtype=torch.bool)
        distance_matrix[self_mask] = torch.max(distance_matrix) + 1
        return labels.float(), distance_matrix

    @staticmethod
    def prepare_labels(**targets) -> torch.Tensor:
        """Compute metric labels

        Args:
            **targets: objects to compute final labels. `**targets` in PairMetric consists of
                `labels`, `pairs` and `subgroups`, in GroupMetric - of `groups`.
        Returns:
            targets: torch.Tensor -  labels to be used during metric computation
        """
        raise NotImplementedError()
