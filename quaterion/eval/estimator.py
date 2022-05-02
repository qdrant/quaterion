import random
from typing import Union, Optional

import torch
import loguru

from quaterion.eval.group import GroupMetric
from quaterion.eval.pair import PairMetric
from quaterion.utils.enums import TrainStage


class Estimator:
    """Class to calculate metrics on whole datasets

    Estimator accumulates embeddings and calculates metric on all, or on fixed-size part of them.
    Estimation might be time and memory consuming operation.
    Estimator can be attached to :class:`quaterion.train.trainable_model.TrainableModel` or used
        on its own.

    Args:
        metric: metric instance for computation
        name: name of estimator for log
        batch_size: determines number of embeddings on which metric should be calculated.
            `-1` corresponds to the whole dataset.
        logger: parameter to be passed directly to model's `log`. Determines whether result should
            be logged into logger like `Tensorboard`, etc.
        policy: if attached to a model, determines a period for estimations (e.g. `3` means
            estimate each 3 epochs). `None` means to estimate only after the end of fitting.
        stage: if attached to a model, determines a stage accumulate embeddings and perform an
            estimation
    """

    def __init__(
        self,
        metric: Union[PairMetric, GroupMetric],
        name: str,
        batch_size: int = -1,
        logger: Optional[bool] = True,
        policy: Optional[int] = None,
        stage: TrainStage = TrainStage.VALIDATION,
    ):
        self.metric = metric
        self.name = name if stage in name else f"{name}_{stage}"
        self.batch_size = batch_size
        self.logger = logger
        self.policy = policy

        self._has_been_reset = True
        loguru.logger.warning(
            f"{metric.compute_on_step} is True in {self.name}. "
            f"It might cause a significant overhead."
        )

    @property
    def has_been_reset(self) -> bool:
        return self._has_been_reset

    def estimate(self) -> torch.Tensor:
        if self.batch_size == -1:
            return self.metric.compute()

        embeddings_num = self.metric.embeddings.shape[0]

        sample_indices = torch.Tensor(
            random.sample(range(embeddings_num), k=max(self.batch_size, embeddings_num))
        )
        return self.metric.compute(sample_indices=sample_indices)

    def update(self, embeddings: torch.Tensor, **targets):
        self._has_been_reset = False
        self.metric.update(embeddings, **targets)

    def reset(self):
        self._has_been_reset = True
        self.metric.reset()
