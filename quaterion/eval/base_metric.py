from torch import Tensor

from quaterion.distances import Distance


class BaseMetric:
    """Base class for evaluation metrics

    Provides a default implementation for distance matrix calculation.

    Args:
        distance_metric_name: name of a distance metric to calculate distance or similarity
            matrices. Available names could be found in :class:`~quaterion.distances.Distance`.

    """

    def __init__(self, distance_metric_name: Distance = Distance.COSINE):
        super().__init__()
        self.distance_metric = Distance.get_by_name(distance_metric_name)
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
        return self.distance_metric.distance_matrix(self.embeddings)

    def calculate_similarities(self) -> Tensor:
        """Calculates similarity matrix

        Returns:
            Tensor: similarity matrix
        """
        return self.distance_metric.similarity_matrix(self.embeddings)
