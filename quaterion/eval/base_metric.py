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
        self._embeddings = []

    @property
    def embeddings(self):
        """Concatenate list of embeddings to Tensor

        Help to avoid concatenating embeddings for each batch during accumulation. Instead,
        concatenate it only on call.

        Returns:
            torch.Tensor: batch of embeddings
        """
        return torch.cat(self._embeddings) if len(self._embeddings) else torch.Tensor()

    def update(self, **kwargs) -> None:
        """Accumulate batch

        Args:
            **kwargs - embeddings and objects required for label calculation. E.g. for
            :class:`~quaterion.eval.pair.pair_metric.PairMetric` it is `labels`, `pairs`,
            `subgroups` and for :class:`~quaterion.eval.group.group_metric.GroupMetric` it is
            `groups`.
        """
        raise NotImplementedError()

    def reset(self):
        """Reset accumulated state

        Use to reset accumulated embeddings, labels
        """
        self._embeddings = []

    def compute(self, *args, **kwargs) -> torch.Tensor:
        """Compute metric value

        Args:
            *args, **kwargs - contain embeddings and targets required to compute metric.

        Returns:
            torch.Tensor - computed metric
        """
        raise NotImplementedError()

    def evaluate(self) -> torch.Tensor:
        """Perform metric computation with accumulated state"""
        raise NotImplementedError()

    def raw_compute(self, distance_matrix: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
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
        labels = self.compute_labels(**targets)
        distance_matrix = self.calculate_distance_matrix(embeddings)
        self_mask = torch.eye(distance_matrix.shape[0], dtype=torch.bool)
        distance_matrix[self_mask] = torch.max(distance_matrix) + 1
        return labels.float(), distance_matrix

    def compute_labels(self, **targets) -> torch.Tensor:
        """Compute metric labels

        Args:
            **targets: objects to compute final labels. `**targets` in PairMetric consists of
                `labels`, `pairs` and `subgroups`, in GroupMetric - of `groups`.
        Returns:
            target: torch.Tensor -  labels to be used during metric computation
        """
        raise NotImplementedError()

    def calculate_distance_matrix(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate distance matrix

        Args:
            embeddings: accumulated embeddings

        Returns:
            distance_matrix: torch.Tensor - Shape: (embeddings, embeddings) - distance matrix
        """
        return self.distance.distance_matrix(embeddings)

    def calculate_similarity_matrix(
        self, embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Calculate similarity matrix

        Args:
            embeddings: accumulated embeddings

        Returns:
            similarity_matrix: torch.Tensor - Shape: (embeddings, embeddings) - similarity
                matrix
        """
        return self.distance.similarity_matrix(embeddings)
