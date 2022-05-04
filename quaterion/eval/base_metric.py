from typing import Optional, Tuple, Dict

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

    def compute(
        self,
        *,
        sample_indices: torch.LongTensor = None,
        embeddings: Optional[torch.Tensor] = None,
        **targets
    ) -> Optional[torch.Tensor]:
        """Compute metric value

        Following cases are supported:
            - compute value without arguments passed (explicit call by user)
            - compute value for compute_on_step behaviour (batch items passed)
            - compute value for estimator's call (sample_indices might be passed to calculate
                metric on part of a dataset)

        Args:
            sample_indices: indices of accumulated values for partial metric computation
            embeddings: embeddings to compute metric without accumulated ones
            **targets: targets to compute metric without accumulated ones

        Returns:
            Optional[torch.Tensor] - computed metric value
        """

        if sample_indices is not None and embeddings is not None:
            raise ValueError(
                "`sample_indices` can't be used with `embeddings`. Put `embeddings` into metric"
                "state and call `compute` with `sample_indices` instead."
            )

        if len(self._embeddings) == 0 and embeddings is None:
            if sample_indices is not None:
                raise ValueError(
                    "sample_indices were passed but there is no accumulated embeddings"
                )
            return None

        embeddings, targets = self.prepare_input(embeddings, **targets)
        return self._compute(embeddings, sample_indices=sample_indices, **targets)

    def reset(self):
        """Reset accumulated state

        Use to reset accumulated embeddings, labels
        """
        self._embeddings = []

    def calculate_distance_matrix(
        self,
        embeddings: torch.Tensor,
        ref_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Calculate distance matrix

        Args:
            embeddings: accumulated embeddings
            ref_embeddings: sample of accumulated embeddings

        Returns:
            distance_matrix: torch.Tensor - Shape: (ref_embeddings, embeddings) - distance matrix
        """

        ref_embeddings = embeddings if ref_embeddings is None else ref_embeddings
        distance_matrix = self.distance.distance_matrix(ref_embeddings, embeddings)

        return distance_matrix

    def calculate_similarity_matrix(
        self, embeddings: torch.Tensor, ref_embeddings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Calculate similarity matrix

        Args:
            embeddings: accumulated embeddings
            ref_embeddings: sample of accumulated embeddings

        Returns:
            similarity_matrix: torch.Tensor - Shape: (ref_embeddings, embeddings) - similarity
                matrix
        """
        ref_embeddings = embeddings if ref_embeddings is None else ref_embeddings
        similarity_matrix = self.distance.similarity_matrix(ref_embeddings, embeddings)

        return similarity_matrix

    def prepare_input(
        self, embeddings: Optional[torch.Tensor], **targets
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Prepare input before computation

        If input haven't been passed, substitute accumulated state.

        Args:
            embeddings: embeddings to evaluate
            targets: objects to compute labels for similarity samples

        Returns:
            embeddings, targets: Tuple[torch.Tensor, Dict[str, torch.Tensor]] - prepared embeddings
                and targets
        """
        raise NotImplementedError()

    def precompute(
        self,
        embeddings: torch.Tensor,
        sample_indices: Optional[torch.LongTensor] = None,
        **targets
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepares data for computation

        Compute distance matrix and final labels based on groups.
        Sample embeddings and labels if metric should be computed only on part of the data.

        Args:
            embeddings: embeddings to compute metric value
            targets: objects to compute final labels
            sample_indices: indices to sample embeddings and labels if metric has to be computed
                on part of the data

        Returns:
            torch.Tensor, torch.Tensor - labels and distance matrix
        """
        labels = self.compute_labels(**targets)

        if sample_indices is not None:
            labels = labels[
                sample_indices
            ]  # shape (sample_indices.shape[0], embeddings.shape[0])
            ref_embeddings = embeddings[sample_indices]  # shape
            # (sample_indices.shape[0], embeddings.shape[1])

            distance_matrix = self.calculate_distance_matrix(
                embeddings, ref_embeddings=ref_embeddings
            )  # shape (ref_embeddings.shape[0], embeddings.shape[0])
            index_matrix = torch.arange(0, embeddings.shape[0]).repeat(
                ref_embeddings.shape[0], 1
            )
            self_mask = index_matrix == sample_indices.view(ref_embeddings.shape[0], 1)
        else:
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

    def _compute(
        self,
        embeddings: torch.Tensor,
        *,
        sample_indices: Optional[torch.Tensor] = None,
        **targets
    ) -> torch.Tensor:
        """Compute metric value

        Directly compute metric value.
        This method should be overridden in implementations of a particular metric.
        All additional logic: embeddings and targets preparations, using of cached result etc.
        should be done outside.

        Args:
            embeddings: embeddings to calculate metrics on
            sample_indices: indices of embeddings to sample if metric should be computed only on
                part of accumulated embeddings
            **targets: dict of objects to compute final labels

        Returns:
            torch.Tensor - computed metric
        """
        raise NotImplementedError()
