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

    def __init__(
        self,
        distance_metric_name: Distance = Distance.COSINE,
    ):
        self._labels = []
        self._pairs = []
        self._subgroups = []
        super().__init__(
            distance_metric_name=distance_metric_name,
        )
        self._accumulated_size = 0

    @property
    def labels(self):
        """Concatenate list of labels to Tensor

        Help to avoid concatenating labels for each batch during accumulation. Instead,
        concatenate it only on call.

        Returns:
            torch.Tensor: batch of labels
        """
        return torch.cat(self._labels) if len(self._labels) else torch.Tensor()

    @property
    def subgroups(self):
        """Concatenate list of subgroups to Tensor

        Help to avoid concatenating subgroups for each batch during accumulation. Instead,
        concatenate it only on call.

        Returns:
            torch.Tensor: batch of subgroups
        """
        return torch.cat(self._subgroups) if len(self._subgroups) else torch.Tensor()

    @property
    def pairs(self) -> torch.LongTensor:
        """Concatenate list of pairs to Tensor

        Help to avoid concatenating pairs for each batch during accumulation. Instead,
        concatenate it only on call.

        Returns:
            torch.Tensor: batch of pairs
        """
        return torch.cat(self._pairs) if len(self._pairs) else torch.LongTensor()

    def compute_labels(
        self,
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
            device: device to store calculated embeddings and groups on.
        """
        device = device if device else embeddings.device

        embeddings = embeddings.detach().to(device)
        labels = labels.detach().to(device)
        pairs = pairs.detach().to(device)
        subgroups = subgroups.detach().to(device)

        self._embeddings.append(embeddings)
        self._labels.append(labels)
        self._pairs.append(pairs + self._accumulated_size)
        self._subgroups.append(subgroups)

        self._accumulated_size += pairs.shape[0]

    def reset(self):
        """Reset accumulated state

        Reset embeddings, labels, pairs, subgroups, etc.
        """
        super().reset()
        self._labels = []
        self._pairs = []
        self._subgroups = []
        self._accumulated_size = 0

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
        return self.compute(self.embeddings, self.labels, self.pairs, self.subgroups)

    def raw_compute(
        self, distance_matrix: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError()
