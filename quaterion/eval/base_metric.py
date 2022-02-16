from torch import Tensor


class BaseMetric:
    def eval(self) -> Tensor:
        pass
