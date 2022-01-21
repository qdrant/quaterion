from typing import Type, Callable

from torch import nn, Tensor

from quaterion.loss.metrics import SiameseDistanceMetric


class SimilarityLoss(nn.Module):
    @classmethod
    def metric_class(cls) -> Type:
        return SiameseDistanceMetric

    @classmethod
    def get_distance_function(
        cls, function_name: str
    ) -> Callable[[Tensor, Tensor], Tensor]:
        for name, foo in cls.metric_class().__dict__.items():
            if name == function_name:
                return foo.__func__

        raise RuntimeError(
            f"Unknown `distance_metric_name` {function_name},"
            f" available metrics: {cls.metric_class().__dict__.keys()}"
        )

    def __init__(self, distance_metric_name: str = "cosine_distance"):
        super(SimilarityLoss, self).__init__()
        self.distance_metric_name = distance_metric_name
        self.distance_metric = self.get_distance_function(self.distance_metric_name)

    def get_config_dict(self):
        return {"distance_metric_name": self.distance_metric_name}
