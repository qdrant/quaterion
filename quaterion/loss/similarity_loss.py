from typing import Type, Callable, Dict, Any

from torch import nn, Tensor

from quaterion.loss.metrics import SiameseDistanceMetric


class SimilarityLoss(nn.Module):
    """Base similarity losses class.

    Args:
        distance_metric_name: Name of the function, that returns a distance between two embeddings.
            :class:`~quaterion.loss.metrics.SiameseDistanceMetric` contains pre-defined metrics
            that can be used.
    """

    def __init__(self, distance_metric_name: str = "cosine_distance"):
        super(SimilarityLoss, self).__init__()
        self.distance_metric_name = distance_metric_name
        self.distance_metric = self.get_distance_function(self.distance_metric_name)

    @classmethod
    def metric_class(cls) -> Type:
        """Class with metrics available for current loss.

        Returns:
            Type: class containing metrics
        """
        return SiameseDistanceMetric

    @classmethod
    def get_distance_function(
        cls, function_name: str
    ) -> Callable[[Tensor, Tensor], Tensor]:
        """Retrieve distance function from metric class.

        Args:
            function_name: name of attribute in cls.metric_class instance

        Returns:
            Callable[[Tensor, Tensor], Tensor]: function for distance calculation

        Raises:
            RuntimeError: unknown `distance_metric_name` is passed
        """
        for name, foo in vars(cls.metric_class()).items():
            if name == function_name:
                return foo.__func__

        raise RuntimeError(
            f"Unknown `distance_metric_name` {function_name},"
            f" available metrics: {vars(cls.metric_class()).keys()}"
        )

    def get_config_dict(self) -> Dict[str, Any]:
        """Config used in saving and loading purposes.

        Config object has to be JSON-serializable.

        Returns:
            Dict[str, Any]: JSON-serializable dict of params
        """
        return {"distance_metric_name": self.distance_metric_name}
