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
    """

    def __init__(
        self,
        distance_metric_name: Distance = Distance.COSINE,
    ):
        super().__init__(
            distance_metric_name=distance_metric_name,
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

    def reset(self):
        """Reset accumulated embeddings, groups and cached result"""
        super().reset()
        self._groups = []

    def prepare_input(
        self, embeddings: Optional[Tensor], groups: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Prepare input before computation

        If input haven't been passed, substitute accumulated state (embeddings and groups).

        Args:
            embeddings: embeddings to evaluate
            groups: group numbers to distinguish similar objects from dissimilar

        Returns:
            embeddings, targets: Tuple[torch.Tensor, Dict[str, torch.Tensor]] - prepared embeddings
                and groups dict
        """
        embeddings_passed = embeddings is not None
        targets_passed = groups is not None
        if embeddings_passed != targets_passed:
            raise ValueError(
                "If `embeddings` were passed to `compute`, corresponding `groups` have to be "
                "passed too"
            )

        if not embeddings_passed:
            embeddings = self.embeddings
            groups = self.groups

        return embeddings, {"groups": groups}

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
        groups: Tensor = None
    ):
        """Compute metric value

        Directly compute metric value.
        This method should be overridden in implementations of a particular metric.
        All additional logic: embeddings and targets preparations.
        should be done outside.

        Args:
            embeddings: embeddings to calculate metrics on
            sample_indices: indices of embeddings to sample if metric should be computed only on
                part of accumulated embeddings
            groups: groups to compute final labels

        Returns:
            torch.Tensor - computed metric
        """
        raise NotImplementedError()
