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
        distance_metric_name: Distance = Distance.COSINE,
        compute_on_step=True,
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
    def pairs(self):
        """Concatenate list of pairs to Tensor

        Help to avoid concatenating pairs for each batch during accumulation. Instead,
        concatenate it only on call.

        Returns:
            torch.Tensor: batch of pairs
        """
        return torch.cat(self._pairs) if len(self._pairs) else torch.Tensor()

    def prepare_input(
        self,
        embeddings: Optional[torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        pairs: Optional[torch.LongTensor] = None,
        subgroups: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Prepare input before computation

        If input haven't been passed, substitute accumulated state.

        Args:
            embeddings: embeddings to evaluate
            labels: labels to distinguish similar and dissimilar objects.
            pairs: indices to determine objects of one pair
            subgroups: subgroups numbers to determine which samples can be considered negative

        Returns:
            embeddings, targets: Tuple[torch.Tensor, Dict[str, torch.Tensor]] - prepared embeddings
                and dict with labels, pairs and subgroups to compute final labels
        """
        targets = {}
        embeddings_passed = embeddings is not None
        targets_passed = (
            labels is not None and pairs is not None and subgroups is not None
        )
        if embeddings_passed != targets_passed:
            raise ValueError(
                "If `embeddings` were passed to `compute`, corresponding `labels`, `subgroups` "
                "and `pairs` have to be passed too"
            )

        if not embeddings_passed:
            embeddings = self.embeddings
            labels = self.labels
            pairs = self.pairs
            subgroups = self.subgroups

        targets["labels"] = labels
        targets["pairs"] = pairs
        targets["subgroups"] = subgroups

        return embeddings, targets

    def compute_labels(
        self,
        labels: Optional[torch.Tensor] = None,
        pairs: Optional[torch.LongTensor] = None,
        subgroups: Optional[torch.Tensor] = None,
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
        self._pairs.append(pairs + self._accumulated_size)
        self._subgroups.append(subgroups)

        self._accumulated_size += pairs.shape[0]
        if self.compute_on_step:
            return self.compute(
                embeddings=embeddings, labels=labels, pairs=pairs, subgroups=subgroups
            )

    def reset(self):
        """Reset accumulated state

        Reset embeddings, labels, pairs, subgroups, etc.
        """
        super().reset()
        self._labels = []
        self._pairs = []
        self._subgroups = []
        self._accumulated_size = 0

    def _compute(
        self,
        embeddings: torch.Tensor,
        *,
        sample_indices: Optional[torch.LongTensor] = None,
        labels: Optional[torch.Tensor] = None,
        pairs: Optional[torch.LongTensor] = None,
        subgroups: Optional[torch.Tensor] = None
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
            labels: labels to distinguish similar and dissimilar objects.
            pairs: indices to determine objects of one pair
            subgroups: subgroups numbers to determine which samples can be considered negative

        Returns:
            torch.Tensor - computed metric
        """
        raise NotImplementedError()
