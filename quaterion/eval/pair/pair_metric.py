from typing import Optional, Callable

import torch

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
        compute_on_step=True,
        distance: Distance = Distance.COSINE,
        reduce_func: Optional[Callable] = torch.mean,
    ):
        self._labels = []
        self._pairs = []
        self._subgroups = []
        super().__init__(
            compute_on_step=compute_on_step, distance=distance, reduce_func=reduce_func
        )

    @property
    def labels(self):
        return torch.cat(self._labels) if len(self._labels) else torch.Tensor()

    @property
    def subgroups(self):
        return torch.cat(self._subgroups) if len(self._subgroups) else torch.Tensor()

    @property
    def pairs(self):
        return torch.cat(self._pairs) if len(self._pairs) else torch.Tensor()

    def prepare_input(self, embeddings, targets):
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

    def precompute(self, embeddings, labels, pairs, subgroups, sample_indices=None):

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

    def compute_labels(self, labels=None, pairs=None, _=None):
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

    def update(self, embeddings, labels, pairs, subgroups, device=None):
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
        super().reset()
        self._labels = []
        self._pairs = []
        self._subgroups = []

    def _compute(self, embeddings, *, sample_indices=None, **targets):
        raise NotImplementedError()
