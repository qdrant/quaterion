import random
from typing import Tuple, Sized, Union, Iterable

import torch

from quaterion_models import SimilarityModel

from quaterion.eval.accumulators import GroupAccumulator
from quaterion.eval.group import GroupMetric
from quaterion.eval.samplers import BaseSampler
from quaterion.dataset.similarity_data_loader import GroupSimilarityDataLoader
from quaterion.utils.utils import iter_by_batch
from torch.utils.data import Dataset


class GroupSampler(BaseSampler):
    """Perform selection of embeddings and targets for group based tasks."""

    def __init__(
        self,
        sample_size=-1,
        encode_batch_size=16,
        device: Union[torch.device, str, None] = None,
        log_progress: bool = True,
    ):
        super().__init__(sample_size, device, log_progress)
        self.encode_batch_size = encode_batch_size
        self.accumulator = GroupAccumulator()

    def accumulate(
        self, model: SimilarityModel, dataset: Union[Sized, Iterable, Dataset]
    ):
        """Encodes objects and accumulates embeddings with the corresponding raw labels

        Args:
            model: model to encode objects
            dataset: Sized object, like list, tuple, torch.utils.data.Dataset, etc. to accumulate
        """
        for input_batch in iter_by_batch(
            dataset, self.encode_batch_size, self.log_progress
        ):
            batch_labels = GroupSimilarityDataLoader.collate_labels(input_batch)

            features = [similarity_sample.obj for similarity_sample in input_batch]

            embeddings = model.encode(
                features, batch_size=self.encode_batch_size, to_numpy=False
            )
            self.accumulator.update(embeddings, **batch_labels, device=self.device)

        self.accumulator.set_filled()

    def reset(self):
        """Reset accumulated state"""
        self.accumulator.reset()

    def sample(
        self, dataset: Sized, metric: GroupMetric, model: SimilarityModel
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample embeddings and targets for groups based tasks.

        Args:
            dataset: Sized object, like list, tuple, torch.utils.data.Dataset, etc. to sample
            metric: GroupMetric instance to compute final labels representation
            model: model to encode objects

        Returns:
            torch.Tensor, torch.Tensor: metrics labels and computed distance matrix
        """
        if not self.accumulator.filled:
            self.accumulate(model, dataset)

        embeddings = self.accumulator.embeddings
        labels = metric.prepare_labels(self.accumulator.groups)

        max_sample_size = embeddings.shape[0]

        if self.sample_size > 0:
            sample_size = min(self.sample_size, max_sample_size)
        else:
            sample_size = max_sample_size

        sample_indices = torch.LongTensor(
            random.sample(range(max_sample_size), k=sample_size)
        )
        labels = labels[sample_indices]
        distance_matrix = metric.distance.distance_matrix(
            embeddings[sample_indices], embeddings
        )
        device = embeddings.device
        self_mask = (
            torch.arange(0, distance_matrix.shape[0])
            .view(-1, 1)
            .repeat(1, 2)
            .to(device)
        )
        distance_matrix[self_mask[:, 0], self_mask[:, 1]] = distance_matrix.max() + 1
        return labels.float(), distance_matrix
