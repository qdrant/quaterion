from typing import Union, Optional, Dict

import torch

from quaterion.eval.group import GroupMetric
from quaterion.eval.pair import PairMetric
from quaterion.utils.enums import TrainStage
from quaterion.eval.samplers import BaseSampler


class Evaluator:
    """Class to calculate metrics on whole datasets

    Evaluator accumulates embeddings and calculates metric on all, or on fixed-size part of them.
    Evaluation might be time and memory consuming operation.
    Evaluator can be attached to :class:`quaterion.train.trainable_model.TrainableModel` or used
        on its own.

    Args:
        name: name of evaluator for log
        metric: metric instance for computation
        sampler: sampler selects embeddings and labels to perform partial evaluation
        logger: parameter to be passed directly to model's `log`. Determines whether result should
            be logged into logger like `Tensorboard`, etc.
        epoch_eval_period: if attached to a model, determines a period for evaluations (e.g. `3`
            means estimate each 3 epochs). `None` means to evaluate only after the end of fitting.
        stage: if attached to a model, determines a stage accumulate embeddings and perform an
            estimation
    """

    def __init__(
        self,
        name: str,
        metric: Union[PairMetric, GroupMetric],
        sampler: BaseSampler,
        logger: Optional[bool] = True,
        epoch_eval_period: Optional[int] = None,
        stage: Optional[TrainStage] = TrainStage.VALIDATION,
    ):
        self.metric = metric
        self.name = name if stage in name else f"{name}_{stage}"
        self.sampler = sampler
        self.logger = logger
        self.epoch_eval_period = epoch_eval_period
        self.stage = stage
        self._has_been_reset = True

    @property
    def has_been_reset(self) -> bool:
        return self._has_been_reset

    def evaluate(self) -> torch.Tensor:
        distance_matrix, labels = self.sampler.sample(self.metric)
        return self.metric.raw_compute(distance_matrix, labels)

    def update(self, embeddings: torch.Tensor, **targets):
        self._has_been_reset = False
        self.metric.update(embeddings, **targets)

    def reset(self):
        self._has_been_reset = True
        self.metric.reset()
