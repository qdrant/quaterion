from typing import Sized, Tuple, Union

import torch
from torch import Tensor
from quaterion_models import SimilarityModel

from quaterion.eval.base_metric import BaseMetric


class BaseSampler:
    """Sample part of embeddings and targets to perform metric calculation on a part of the data

    Sampler allows reducing amount of time and resources to calculate a distance matrix.
    Instead of calculation of squared matrix with shape (num_embeddings, num_embeddings), it
    selects embeddings and computes matrix of a rectangle shape.

        Args:
            sample_size: amount of objects to select.

    """

    def __init__(
        self,
        sample_size=-1,
        device: Union[torch.device, str, None] = None,
        log_progress: bool = True,
    ):
        self.log_progress = log_progress
        self.sample_size = sample_size
        self.device = device

    def sample(
        self, dataset: Sized, metric: BaseMetric, model: SimilarityModel
    ) -> Tuple[Tensor, Tensor]:
        """Sample objects and labels to calculate metrics

        Args:
            dataset: Sized object, like list, tuple, torch.utils.data.Dataset, etc. to sample
            metric: metric instance to compute final labels representation
            model: model to encode objects

        Returns:
            Tensor, Tensor: metrics labels and computed distance matrix
        """
        pass

    def reset(self):
        """Reset accumulated state if any"""
        pass
