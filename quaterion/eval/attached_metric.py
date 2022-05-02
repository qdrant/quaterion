from typing import Union

from quaterion.eval.group import GroupMetric
from quaterion.eval.pair import PairMetric


class AttachedMetric:
    def __init__(
        self, name, metric: Union[PairMetric, GroupMetric], stages, **log_options
    ):
        self._metric = metric
        self.stages = stages
        self.name = name
        self.log_options = log_options

    def __getattr__(self, item):
        try:
            return getattr(self._metric, item)
        except AttributeError as ae:
            raise AttributeError(
                f"`BuiltinMetric` object (<{self.name}>) has no attribute {item}"
            ) from ae
