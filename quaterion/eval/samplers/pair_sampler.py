import random
from typing import Tuple, Union
from collections.abc import Sized

import torch

from quaterion.eval.accumulators import PairAccumulator
from quaterion.eval.pair import PairMetric
from quaterion.eval.samplers import BaseSampler
from quaterion_models import SimilarityModel
from quaterion.dataset.similarity_data_loader import PairsSimilarityDataLoader
from quaterion.utils.utils import iter_by_batch


class PairSampler(BaseSampler):
    """Perform selection of embeddings and targets for pairs based tasks.

    Sampler allows reducing amount of time and resources to calculate a distance matrix.
    Instead of calculation of squared matrix with shape (num_embeddings, num_embeddings), it
    selects embeddings and computes matrix of a rectangle shape.

    Args:
        sample_size: int - amount of objects to select
        distinguish: bool - determines whether to compare all objects each-to-each, or to
            compare only `obj_a` to `obj_b`. If true - compare only `obj_a` to `obj_b`.
            Significantly reduces matrix size.
        encode_batch_size: int - batch size to use during encoding

    """

    def __init__(
        self,
        sample_size: int = -1,
        distinguish: bool = False,
        encode_batch_size: int = 16,
        device: Union[torch.device, str, None] = None,
        log_progress: bool = True,
    ):
        super().__init__(sample_size, device, log_progress)
        self.encode_batch_size = encode_batch_size
        self.distinguish = distinguish
        self.accumulator = PairAccumulator()

    def accumulate(self, model: SimilarityModel, dataset: Sized):
        """Encodes objects and accumulates embeddings with the corresponding raw labels

        Args:
            model: model to encode objects
            dataset: Sized object, like list, tuple, torch.utils.data.Dataset, etc. to accumulate
        """
        for input_batch in iter_by_batch(
            dataset, self.encode_batch_size // 2, self.log_progress
        ):
            batch_labels = PairsSimilarityDataLoader.collate_labels(input_batch)

            objects_a, objects_b = [], []
            for similarity_sample in input_batch:
                objects_a.append(similarity_sample.obj_a)
                objects_b.append(similarity_sample.obj_b)

            features = objects_a + objects_b
            embeddings = model.encode(
                features, batch_size=self.encode_batch_size, to_numpy=False
            )
            self.accumulator.update(embeddings, **batch_labels, device=self.device)

        self.accumulator.set_filled()

    def reset(self):
        """Reset accumulated state"""
        self.accumulator.reset()

    def sample(
        self, dataset: Sized, metric: PairMetric, model: SimilarityModel
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample embeddings and targets for pairs based tasks.

        Args:
            dataset: Sized object, like list, tuple, torch.utils.data.Dataset, etc. to sample
            metric: PairMetric instance to compute final labels representation
            model: model to encode objects

        Returns:
            torch.Tensor, torch.Tensor: metrics labels and computed distance matrix
        """
        if not self.accumulator.filled:
            self.accumulate(model, dataset)

        embeddings = self.accumulator.embeddings
        pairs = self.accumulator.pairs

        labels = metric.prepare_labels(
            self.accumulator.labels, pairs, self.accumulator.subgroups
        )

        embeddings_num = embeddings.shape[0]
        max_sample_size = embeddings_num if not self.distinguish else pairs.shape[0]

        if self.sample_size > 0:
            sample_size = min(self.sample_size, max_sample_size)
        else:
            sample_size = max_sample_size

        sample_indices = torch.LongTensor(
            random.sample(range(max_sample_size), k=sample_size)
        )
        labels = labels[sample_indices]

        if self.distinguish:
            ref_embeddings = embeddings[pairs[sample_indices][:, 0]]
            embeddings = embeddings[pairs[:, 1]]
            labels = labels[:, pairs[:, 1]]
            distance_matrix = metric.distance.distance_matrix(
                ref_embeddings, embeddings
            )
        else:
            ref_embeddings = embeddings[sample_indices]
            distance_matrix = metric.distance.distance_matrix(
                ref_embeddings, embeddings
            )
            device = embeddings.device
            self_mask = (
                torch.arange(
                    0, distance_matrix.shape[0], dtype=torch.long, device=device
                )
                .view(-1, 1)
                .to(device)
            )
            self_mask = torch.cat([self_mask, sample_indices.view(-1, 1)], dim=1)
            distance_matrix[self_mask[:, 0], self_mask[:, 1]] = (
                distance_matrix.max() + 1
            )

        return labels.float(), distance_matrix
