from typing import Optional, Callable

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

    def __init__(
        self,
        compute_on_step=True,
        distance: Distance = Distance.COSINE,
        reduce_func: Optional[Callable] = torch.mean,
    ):
        super().__init__(
            compute_on_step=compute_on_step, distance=distance, reduce_func=reduce_func
        )
        self._groups = []

    @property
    def groups(self):
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
            self.compute(embeddings, groups=groups)

    def reset(self):
        """Reset accumulated embeddings and groups"""
        super().reset()
        self._groups = []

    def precompute(self, embeddings, groups, sample_indices=None):

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

    def prepare_input(self, embeddings, targets):
        embeddings_passed = embeddings is not None
        targets_passed = bool(targets)
        if embeddings_passed != targets_passed:
            raise ValueError(
                "If `embeddings` were passed to `compute`, corresponding `groups` have to be "
                "passed too"
            )

        if not embeddings_passed:
            embeddings = self.embeddings
            targets["groups"] = self._groups

        return embeddings, targets

    def compute_labels(self, groups=None):
        if groups is None:
            groups = self.groups

        group_matrix = groups.repeat(groups.shape[0], 1)
        # objects with the same groups are true, others are false

        group_mask = (group_matrix == groups.unsqueeze(1)).bool()
        # exclude obj
        group_mask[torch.eye(group_mask.shape[0], dtype=torch.bool)] = False
        return group_mask

    def _compute(self, embeddings, *, sample_indices=None, **target):
        raise NotImplementedError()
