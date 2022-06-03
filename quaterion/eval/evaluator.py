from typing import Dict, Sized, Union, Iterable

import torch
from torch.utils.data import Dataset
from quaterion_models import SimilarityModel

from quaterion.eval.samplers import BaseSampler
from quaterion.eval.base_metric import BaseMetric


class Evaluator:
    """Calculate metrics on the whole datasets

    Calculates metric on the whole dataset or on sampled part of it.
    Evaluation might be time and memory consuming operation.

    Args:
        metrics: dictionary of metrics instances for calculation
        sampler: sampler selects embeddings and labels to perform partial evaluation
    """

    def __init__(
        self,
        metrics: Union[BaseMetric, Dict[str, BaseMetric]],
        sampler: BaseSampler,
    ):
        self.metrics = (
            metrics
            if isinstance(metrics, dict)
            else {metrics.__class__.__name__: metrics}
        )
        self.sampler = sampler

    def evaluate(
        self,
        dataset: Union[Sized, Iterable, Dataset],
        model: SimilarityModel,
    ) -> Dict[str, torch.Tensor]:
        """Compute metrics on a dataset

        Args:
            dataset: Sized object, like list, tuple, torch.utils.data.Dataset, etc. to compute
                metrics
            model: SimilarityModel instance to perform objects encoding

        Returns:
            Dict[str, torch.Tensor] - dict of computed metrics
        """
        results = {}
        for metric_name, metric in self.metrics.items():
            labels, distance_matrix = self.sampler.sample(dataset, metric, model)
            results[metric_name] = metric.raw_compute(distance_matrix, labels)

        self.sampler.reset()
        return results
