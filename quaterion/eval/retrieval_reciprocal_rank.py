import torch

from quaterion.eval.base_metric import BaseMetric


class RetrievalReciprocalRank(BaseMetric):
    def __init__(self, distance_metric, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.result = torch.nan

    def update(self, predictions: torch.Tensor, target: torch.Tensor):
        indices = torch.argsort(predictions, dim=1, descending=True)
        target = target.gather(1, indices)
        position = torch.nonzero(target)
        self.result = 1.0 / (position[:, 1] + 1.0)
        return self.result

    def compute(self):
        return self.result
