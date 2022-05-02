from typing import Optional, Callable, Dict, Tuple

import torch

from quaterion.distances import Distance
from quaterion.eval.base_metric import BaseMetric


class PairMetric(BaseMetric):
    """Base class for metrics computation for pair based data

    Args:
        distance_metric_name: name of a distance metric to calculate distance or similarity
            matrices. Available names could be found in :class:`~quaterion.distances.Distance`.
        compute_on_step: flag if metric should be calculated on each batch
        reduce_func: function to reduce calculated metric. E.g. `torch.mean`, `torch.max` and
            others. `functools.partial` might be useful if you want to capture some custom
            arguments.

    Provides default implementations for embeddings and labels accumulation, distance and
    interaction matrices calculation.
    """

    def __init__(
        self,
        compute_on_step=True,
        distance_metric_name: Distance = Distance.COSINE,
        reduce_func: Optional[Callable] = torch.mean,
    ):
        self._labels = []
        self._pairs = []
        self._subgroups = []
        super().__init__(
            compute_on_step=compute_on_step,
            distance_metric_name=distance_metric_name,
            reduce_func=reduce_func,
        )

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
    def pairs(self):
        """Concatenate list of pairs to Tensor

        Help to avoid concatenating pairs for each batch during accumulation. Instead,
        concatenate it only on call.

        Returns:
            torch.Tensor: batch of pairs
        """
        return torch.cat(self._pairs) if len(self._pairs) else torch.Tensor()

    def prepare_input(
        self, embeddings: Optional[torch.Tensor], **targets
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Prepare input before computation

        If input haven't been passed, substitute accumulated state.

        Args:
            embeddings: embeddings to evaluate
            targets: labels, pairs and subgroups to compute final labels

        Returns:
            embeddings, targets: Tuple[torch.Tensor, Dict[str, torch.Tensor]] - prepared embeddings
                and dict with labels, pairs and subgroups to compute final labels
        """
        embeddings_passed = embeddings is not None
        targets_passed = bool(targets)
        if embeddings_passed != targets_passed:
            raise ValueError(
                "If `embeddings` were passed to `compute`, corresponding `labels`, `subgroups` "
                "and `pairs` have to be passed too"
            )

        if not embeddings_passed:
            embeddings = self.embeddings
            targets["labels"] = self.labels
            targets["pairs"] = self.pairs
            targets["subgroups"] = self.subgroups

        return embeddings, targets

    def precompute(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        pairs: torch.LongTensor,
        subgroups: torch.Tensor,
        sample_indices: Optional[torch.Tensor] = None,
    ):
        """Prepares data for computation

        Compute distance matrix and final labels based on labels and pairs.
        Sample embeddings and labels if metric should be computed only on part of the data.

        Args:
            embeddings: embeddings to compute metric value
            labels: labels to distinguish similar and dissimilar objects
            pairs: indices to determine objects of one pair
            subgroups: subgroups numbers to determine which samples can be considered negative
            sample_indices: indices to sample embeddings and labels if metric has to be computed
                on part of the data

        Returns:
            torch.Tensor, torch.Tensor - labels and distance matrix
        """
        labels = self.compute_labels(labels, pairs, subgroups)
        if sample_indices is not None:
            labels = labels[sample_indices]
            ref_embeddings = embeddings[sample_indices]
            distance_matrix = self.calculate_distance_matrix(ref_embeddings, embeddings)
            self_mask = sample_indices
        else:
            distance_matrix = self.calculate_distance_matrix(embeddings)
            self_mask = torch.eye(distance_matrix.shape[1], dtype=torch.bool)

        distance_matrix[self_mask] = torch.max(distance_matrix) + 1
        return labels, distance_matrix

    def compute_labels(self, labels=None, pairs=None, _=None) -> torch.Tensor:
        """Compute metric labels based on samples labels and pairs

        Args:
            labels: labels to distinguish similar and dissimilar objects
            pairs: indices to determine objects belong to the same pair
            _: subgroups. Currently, they are not used for labels computation

        Returns:
            target: torch.Tensor -  labels to be used during metric computation
        """
        if labels is None or pairs is None:
            pairs = self.pairs
            labels = self.labels

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
        self._pairs.append(pairs)
        self._subgroups.append(subgroups)
        self._updated = True

        if self.compute_on_step:
            return self.compute(
                embeddings=embeddings, labels=labels, pairs=pairs, subgroups=subgroups
            )

    def reset(self):
        """Reset accumulated state

        Reset embeddings, labels, pairs, subgroups and cached result.
        """
        super().reset()
        self._labels = []
        self._pairs = []
        self._subgroups = []

    def _compute(
        self,
        embeddings: torch.Tensor,
        *,
        sample_indices: Optional[torch.Tensor] = None,
        **targets
    ):
        """Compute metric value

        Directly compute metric value.
        This method should be overridden in implementations of a particular metric.
        All additional logic: embeddings and targets preparations, using of cached result etc.
        should be done outside.

        Args:
            embeddings: embeddings to calculate metrics on
            sample_indices: indices of embeddings to sample if metric should be computed only on
                part of accumulated embeddings
            **targets: labels, pairs and subgroups to compute final labels

        Returns:
            torch.Tensor - computed metric
        """
        raise NotImplementedError()
