from typing import Optional, Callable, Dict, Tuple

import torch
from torch import Tensor, LongTensor

from quaterion.distances import Distance
from quaterion.eval.base_metric import BaseMetric


class GroupMetric(BaseMetric):
    """Base class for group metrics

    Provide default implementation for embeddings and groups accumulation.

    Args:
        distance_metric_name: name of a distance metric to calculate distance or similarity
            matrices. Available names could be found in :class:`~quaterion.distances.Distance`.
        compute_on_step: flag if metric should be calculated on each batch
        reduce_func: function to reduce calculated metric. E.g. `torch.mean`, `torch.max` and
            others. `functools.partial` might be useful if you want to capture some custom
            arguments.
    """

    def __init__(
        self,
        distance_metric_name: Distance = Distance.COSINE,
        compute_on_step=True,
        reduce_func: Optional[Callable] = torch.mean,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            distance_metric_name=distance_metric_name,
            reduce_func=reduce_func,
        )
        self._groups = []

    @property
    def groups(self):
        """Concatenate list of groups to Tensor

        Help to avoid concatenating groups for each batch during accumulation. Instead,
        concatenate it only on call.

        Returns:
            torch.Tensor: batch of groups
        """
        return torch.cat(self._groups) if len(self._groups) else torch.Tensor()

    def update(self, embeddings: Tensor, groups: torch.LongTensor, device=None) -> None:
        """Process and accumulate batch

        Args:
            embeddings: embeddings to accumulate
            groups: groups to distinguish similar and dissimilar objects.
            device: device to store calculated embeddings and groups on.
        """

        if device is None:
            device = embeddings.device

        embeddings = embeddings.detach().to(device)
        groups = groups.detach().to(device)

        self._embeddings.append(embeddings)
        self._groups.append(groups)
        self._updated = True

        if self.compute_on_step:
            self.compute(embeddings=embeddings, groups=groups)

    def reset(self):
        """Reset accumulated embeddings, groups and cached result"""
        super().reset()
        self._groups = []

    def precompute(
        self,
        embeddings: Tensor,
        groups: Tensor,
        sample_indices: Optional[LongTensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Prepares data for computation

        Compute distance matrix and final labels based on groups.
        Sample embeddings and labels if metric should be computed only on part of the data.

        Args:
            embeddings: embeddings to compute metric value
            groups: groups to distinguish similar and dissimilar objects
            sample_indices: indices to sample embeddings and labels if metric has to be computed
                on part of the data

        Returns:
            torch.Tensor, torch.Tensor - labels and distance matrix
        """
        labels = self.compute_labels(groups)

        if sample_indices:
            labels = labels[sample_indices]
            ref_embeddings = embeddings[sample_indices]
            distance_matrix = self.calculate_distance_matrix(ref_embeddings, embeddings)
        else:
            distance_matrix = self.calculate_distance_matrix(embeddings)

        distance_matrix[torch.eye(distance_matrix.shape[1], dtype=torch.bool)] = (
            torch.max(distance_matrix) + 1
        )
        return labels.float(), distance_matrix

    def prepare_input(
        self, embeddings: Optional[Tensor], **targets
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Prepare input before computation

        If input haven't been passed, substitute accumulated state (embeddings and groups).

        Args:
            embeddings: embeddings to evaluate
            targets: groups

        Returns:
            embeddings, targets: Tuple[torch.Tensor, Dict[str, torch.Tensor]] - prepared embeddings
                and groups dict
        """
        embeddings_passed = embeddings is not None
        targets_passed = bool(targets)
        if embeddings_passed != targets_passed:
            raise ValueError(
                "If `embeddings` were passed to `compute`, corresponding `groups` have to be "
                "passed too"
            )

        if not embeddings_passed:
            embeddings = self.embeddings
            targets["groups"] = self.groups

        return embeddings, targets

    def compute_labels(self, groups: Optional[Tensor] = None):
        """Compute metric labels based on samples groups

        Args:
            groups: groups to distinguish similar and dissimilar objects

        Returns:
            target: torch.Tensor -  labels to be used during metric computation
        """
        if groups is None:
            groups = self.groups

        group_matrix = groups.repeat(groups.shape[0], 1)
        # objects with the same groups are true, others are false

        group_mask = (group_matrix == groups.unsqueeze(1)).bool()
        # exclude obj
        group_mask[torch.eye(group_mask.shape[0], dtype=torch.bool)] = False
        return group_mask

    def _compute(
        self,
        embeddings: Tensor,
        *,
        sample_indices: Optional[LongTensor] = None,
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
            **targets: groups to compute final labels

        Returns:
            torch.Tensor - computed metric
        """
        raise NotImplementedError()
