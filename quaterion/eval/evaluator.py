from typing import Dict

import torch
from torch.utils.data import Dataset

from quaterion.eval.samplers import BaseSampler
from quaterion.eval.base_metric import BaseMetric
from quaterion_models import MetricModel


class Evaluator:
    """Class to calculate metrics on whole datasets

    Calculates metric on the whole dataset or on sampled part of it.
    Evaluation might be time and memory consuming operation.

    Args:
        metric: metric instance for computation
        sampler: sampler selects embeddings and labels to perform partial evaluation
        dataset: dataset instance to evaluate
    """

    def __init__(
        self,
        metrics: Dict[str, BaseMetric],
        sampler: BaseSampler,
        model: MetricModel,
    ):
        self.metrics = metrics
        self.sampler = sampler
        self.model = model

    def evaluate(self, dataset: Dataset) -> Dict[str, torch.Tensor]:

        results = {}
        for metric_name, metric in self.metrics.items():

            distance_matrix, labels = self.sampler.sample(
                dataset,
                metric,
                self.model
            )
            results[metric_name] = metric.raw_compute(distance_matrix, labels)

        self.sampler.reset()
        return results
