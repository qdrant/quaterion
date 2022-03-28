import torch

from quaterion.eval.base_metric import BaseMetric


class RetrievalPrecision(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.result = torch.nan

    def update(self, predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.result = target.gather(1, predictions.topk(1, dim=1)[1]).sum(dim=1).float()
        return self.result

    def compute(self):
        return self.result
