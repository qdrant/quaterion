from quaterion.utils.enums import TrainStage
from quaterion.eval.base_metric import BaseMetric


class AttachedMetric:
    """Class to attach metric to :class:`~quaterion.train.trainable_model.TrainableModel`

    Contain required parameters to compute and log batch-wise metric during training process.

    Args:
        name: name of an attached metric to be used in log.
        metric: metric to be calculated.
        **log_options: additional kwargs to be passed to model's log.

        The most often `log_options` keys are:
            on_step: Logs the metric at the current step.
            on_epoch: Automatically accumulates and logs at the end of the epoch.
            prog_bar: Logs to the progress bar (Default: False).
            logger: Logs to the logger like Tensorboard, or any other custom logger passed to the
                Trainer (Default: True).

        The remaining options can be found at:
        https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html
    """

    def __init__(
        self,
        name: str,
        metric: BaseMetric,
        **log_options,
    ):
        self._metric = metric
        self.stages = [TrainStage.TRAIN, TrainStage.VALIDATION]
        self.name = name
        self.log_options = log_options

    def __getattr__(self, item: str):
        try:
            return getattr(self._metric, item)
        except AttributeError as ae:
            raise AttributeError(
                f"`AttachedMetric` object (<{self.name}>) has no attribute {item}"
            ) from ae
