import torch
from torch import Tensor, LongTensor

from quaterion.distances import Distance
from quaterion.eval.accumulators import PairAccumulator
from quaterion.eval.base_metric import BaseMetric


class PairMetric(BaseMetric):
    """Base class for metrics computation for pair based data

    Args:
        distance_metric_name: name of a distance metric to calculate distance or similarity
            matrices. Available names could be found in :class:`~quaterion.distances.Distance`.

    Provides default implementations for distance and interaction matrices calculation.
    Accumulates embeddings and labels in an accumulator.
    """

    def __init__(
        self,
        distance_metric_name: Distance = Distance.COSINE,
    ):
        super().__init__(
            distance_metric_name=distance_metric_name,
        )
        self.accumulator = PairAccumulator()

    @staticmethod
    def prepare_labels(
        labels: torch.Tensor,
        pairs: torch.LongTensor,
        subgroups: torch.Tensor,
    ) -> torch.Tensor:
        """Compute metric labels based on samples labels and pairs

        Args:
            labels: labels to distinguish similar and dissimilar objects
            pairs: indices to determine objects belong to the same pair
            subgroups: indices to determine negative examples. Currently, they are not used for
                labels computation.

        Returns:
            target: torch.Tensor -  labels to be used during metric computation
        """
        num_of_embeddings = pairs.shape[0] * 2
        target = torch.zeros(
            (num_of_embeddings, num_of_embeddings), device=labels.device
        )
        # todo: subgroups should also be taken into account
        target[pairs[:, 0], pairs[:, 1]] = labels
        target[pairs[:, 1], pairs[:, 0]] = labels
        return target

    def reset(self):
        """Reset accumulated state

        Reset embeddings, labels, pairs, subgroups, etc.
        """
        self.accumulator.reset()

    def compute(
        self, embeddings: Tensor, labels: Tensor, pairs: LongTensor, subgroups: Tensor
    ):
        """Compute metric value

        Args:
            embeddings: embeddings to calculate metrics on
            labels: labels to distinguish similar and dissimilar objects.
            pairs: indices to determine objects of one pair
            subgroups: subgroups numbers to determine which samples can be considered negative

        Returns:
            torch.Tensor - computed metric
        """
        labels, distance_matrix = self.precompute(
            embeddings, labels=labels, pairs=pairs, subgroups=subgroups
        )
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
            labels: labels ready to metric computation with the same shape as `distance_matrix`.
                Values are taken from `SimilarityPairSample.score`.

        Returns:
            torch.Tensor - calculated metric value
        """
        raise NotImplementedError()

    def update(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        pairs: torch.LongTensor,
        subgroups: torch.Tensor,
        device=None,
    ):
        """Process and accumulate batch

        Args:
            embeddings: embeddings to accumulate
            labels: labels to distinguish similar and dissimilar objects.
            pairs: indices to determine objects of one pair
            subgroups: subgroups numbers to determine which samples can be considered negative
            device: device to store calculated embeddings and targets on.
        """
        self.accumulator.update(embeddings, labels, pairs, subgroups, device)
