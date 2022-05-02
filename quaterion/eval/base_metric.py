from typing import Callable, Optional, Tuple, Dict

import torch

from quaterion.distances import Distance


class BaseMetric:
    """Base class for evaluation metrics

    Provides a default implementation for distance matrix calculation.

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
        self.distance = Distance.get_by_name(distance_metric_name)
        self.compute_on_step = compute_on_step

        self._reduce_func = reduce_func
        self._distance_metric_name = distance_metric_name
        self._embeddings = []
        self._cached_result = None
        self._updated = True

    @property
    def embeddings(self):
        """Concatenate list of embeddings to Tensor

        Help to avoid concatenating embeddings for each batch during accumulation. Instead,
        concatenate it only on call.

        Returns:
            torch.Tensor: batch of embeddings
        """
        return torch.cat(self._embeddings) if len(self._embeddings) else torch.Tensor()

    def compute(
        self,
        sample_indices: torch.Tensor = None,
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

        if not self._updated and embeddings is None:
            return self._cached_result

        embeddings, targets = self.prepare_input(embeddings, **targets)

        raw_value = self._compute(embeddings, sample_indices=sample_indices, **targets)

        if self._reduce_func is None or raw_value is None:
            return raw_value

        return self._reduce_func(raw_value)

    def reset(self):
        """Reset accumulated state

        Use to reset accumulated embeddings, labels and cached result
        """
        self._cached_result = None
        self._embeddings = []

    def calculate_distance_matrix(
        self, embeddings: torch.Tensor, ref_embeddings: Optional[torch.Tensor] = None,
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
