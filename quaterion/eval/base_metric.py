from typing import Callable

from torch import Tensor


class BaseMetric:
    """Base class for evaluation metrics

    Provides a default implementation for distance matrix calculation.

    Args:
        distance_metric: function for distance matrix computation. Possible choice might be one of
            :class:`~quaterion.loss.metrics.SiameseDistanceMetric` methods.

    """

    def __init__(self, distance_metric: Callable):
        super().__init__()
        self.distance_metric = distance_metric
        self.embeddings = Tensor()

    def compute(self) -> Tensor:
        """Calculates metric

        Returns:
            Tensor: metric result
        """
        raise NotImplementedError()

    def calculate_distances(self) -> Tensor:
        """Calculates distance matrix

        Returns:
            Tensor: distance matrix
        """
        return self.distance_metric(self.embeddings, self.embeddings, matrix=True)
