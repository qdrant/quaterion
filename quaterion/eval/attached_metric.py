from typing import Union, List

from quaterion.utils.enums import TrainStage

from quaterion.eval.group import GroupMetric
from quaterion.eval.pair import PairMetric


class AttachedMetric:
    """Class to attach metric to :class:`~quaterion.train.trainable_model.TrainableModel`

    Contain required parameters to compute and log batch-wise metric during training process.

    Args:
        name: name of an attached metric to be used in log.
        metric: metric to be calculated.
        stages: stages to calculate metric on. Training, validation, etc.
        **log_options: additional kwargs to be passed to model's log.
    """

    def __init__(
        self,
        name: str,
        metric: Union[PairMetric, GroupMetric],
        stages: Union[TrainStage, List[TrainStage]],
        **log_options,
    ):
        if not metric.compute_on_step:
            raise ValueError("`metric.compute_on_step` must be `True` in `AttachedMetric`")

        self._metric = metric
        self.stages = [stages] if isinstance(stages, TrainStage) else stages
        self.name = name
        self.log_options = log_options

    def __getattr__(self, item: str):
        try:
            return getattr(self._metric, item)
        except AttributeError as ae:
            raise AttributeError(
                f"`AttachedMetric` object (<{self.name}>) has no attribute {item}"
            ) from ae
