from typing import Any, Dict

from quaterion.distances import Distance
from torch import nn


class SimilarityLoss(nn.Module):
    """Base similarity losses class.

    Args:
        distance_metric_name: Name of the distance function, e.g.,
            :class:`~quaterion.distances.Distance`.
    """

    def __init__(self, distance_metric_name: Distance = Distance.COSINE):
        super(SimilarityLoss, self).__init__()
        self.distance_metric = Distance.get_by_name(distance_metric_name)
        self.distance_metric_name = distance_metric_name

    def get_config_dict(self) -> Dict[str, Any]:
        """Config used in saving and loading purposes.

        Config object has to be JSON-serializable.

        Returns:
            Dict[str, Any]: JSON-serializable dict of params
        """
        return {"distance_metric_name": self.distance_metric_name}
