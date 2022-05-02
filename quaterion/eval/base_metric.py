from typing import Callable, Optional

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
        distance: Distance = Distance.COSINE,
        compute_on_step=True,
        reduce_func: Optional[Callable] = torch.mean,
    ):
        self.distance = Distance.get_by_name(distance)
        self.compute_on_step = compute_on_step

        self._reduce_func = reduce_func
        self._distance_name = distance
        self._embeddings = []
        self._cached_result = None
        self._updated = True

    @property
    def embeddings(self):
        return torch.cat(self._embeddings)

    def compute(self, sample_indices=None, embeddings=None, **targets):
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

        embeddings, targets = self.prepare_input(embeddings, targets)

        raw_value = self._compute(embeddings, sample_indices=sample_indices, **targets)

        if self._reduce_func is None or raw_value is None:
            return raw_value

        return self._reduce_func(raw_value)

    def reset(self):
        self._cached_result = None
        self._embeddings = []

    def calculate_distance_matrix(
        self, embeddings, ref_embeddings=None,
    ):
        if ref_embeddings is None:
            distance_matrix = self.distance.distance_matrix(embeddings)
        else:
            distance_matrix = self.distance.distance_matrix(ref_embeddings, embeddings)

        return distance_matrix

    def calculate_similarity_matrix(self, embeddings, ref_embeddings=None):
        if ref_embeddings is None:
            similarity_matrix = self.distance.similarity_matrix(embeddings)
        else:
            similarity_matrix = self.distance.similarity_matrix(
                ref_embeddings, embeddings
            )

        return similarity_matrix

    def prepare_input(self, embeddings, targets):
        raise NotImplementedError()

    def _compute(self, embeddings, *, sample_indices=None, **targets):
        raise NotImplementedError()
