from typing import Dict, Sized

import torch
from quaterion_models import MetricModel

from quaterion.eval.samplers import BaseSampler
from quaterion.eval.base_metric import BaseMetric


class Evaluator:
    """Class to calculate metrics on whole datasets

    Calculates metric on the whole dataset or on sampled part of it.
    Evaluation might be time and memory consuming operation.

    Args:
        metrics: dictionary of metrics instances for calculation
        sampler: sampler selects embeddings and labels to perform partial evaluation
        model: metric model instance to perform objects encoding
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

    def evaluate(self, dataset: Sized) -> Dict[str, torch.Tensor]:
        """Compute metrics on a dataset

        Args:
            dataset: Sized object, like list, tuple, torch.utils.data.Dataset, etc. to compute
                metrics

        Returns:
            Dict[str, torch.Tensor] - dict of computed metrics
        """
        results = {}
        for metric_name, metric in self.metrics.items():

            distance_matrix, labels = self.sampler.sample(dataset, metric, self.model)
            results[metric_name] = metric.raw_compute(distance_matrix, labels)

        self.sampler.reset()
        return results
