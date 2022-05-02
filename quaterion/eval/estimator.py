from typing import Union

import torch

from quaterion.eval.group import GroupMetric
from quaterion.eval.pair import PairMetric
from quaterion.utils.enums import TrainStage


class Estimator:
    def __init__(
        self,
        metric: Union[PairMetric, GroupMetric],
        name,
        batch_size=-1,
        logger=None,
        policy=None,
        stage=TrainStage.VALIDATION,
    ):
        self.metric = metric
        self.name = name if stage in name else f"{name}_{stage}"
        self.batch_size = batch_size
        self.logger = logger
        self.policy = policy

        self._has_been_reset = True
        # todo: log a warning if metric.compute_on_step is True

    @property
    def has_been_reset(self):
        return self._has_been_reset

    def estimate(self):
        if self.batch_size == -1:
            return self.metric.compute()

        embeddings_num = self.metric.embeddings.shape[0]
        # todo: remove replacement
        sample_indices = torch.randint(high=embeddings_num, size=(max(self.batch_size, embeddings_num),))
        return self.metric.compute(sample_indices=sample_indices)

    def update(self, embeddings, **targets):
        self._has_been_reset = False
        self.metric.update(embeddings, **targets)

    def reset(self):
        self._has_been_reset = True
        self.metric.reset()
