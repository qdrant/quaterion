import random
from typing import Tuple

import torch

from quaterion.eval.accumulators import PairAccumulator
from quaterion.eval.pair import PairMetric
from quaterion.eval.samplers import BaseSampler
from torch.utils.data import Dataset, DataLoader
from quaterion_models import MetricModel
from quaterion.dataset.similarity_data_loader import PairsSimilarityDataLoader


class PairSampler(BaseSampler):
    """Perform selection of embeddings and targets for pairs based tasks.

    Args:
        distinguish: bool - determines whether to compare all objects each-to-each, or to
            compare only `obj_a` to `obj_b`. If true - compare only `obj_a` to `obj_b`. Reduces
            matrix size quadratically.

    """

    def __init__(self, sample_size=-1, distinguish=False, encode_batch_size=16):
        super().__init__(sample_size)
        self.encode_batch_size = encode_batch_size
        self.distinguish = distinguish
        self.accumulator = PairAccumulator()

    def accumulate(self, dataset, model):
        dataloader = DataLoader(dataset, batch_size=self.encode_batch_size)
        collate_labels = PairsSimilarityDataLoader.collate_labels

        for batch in dataloader:
            objects = []
            # preserve the same order as in `collate_labels`
            for similarity_pair_sample in batch:
                objects.append(similarity_pair_sample.obj_a)

            for similarity_pair_sample in batch:
                objects.append(similarity_pair_sample.obj_b)

            batch_labels = collate_labels(batch)
            self.accumulator.update(model.encode(objects), **batch_labels)

        self.accumulator.set_filled()

    def reset(self):
        self.accumulator.reset()

    def sample(
        self, dataset: Dataset, metric: PairMetric, model: MetricModel
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample embeddings and targets for pairs based tasks.

        Args:
            dataset: ...
            metric: PairMetric instance with accumulated embeddings, labels, pairs and subgroups
            model: ...

        Returns:
            torch.Tensor, torch.Tensor: metrics labels and computed distance matrix
        """
        if not self.accumulator.filled:
            self.accumulate(dataset, model)

        embeddings = self.accumulator.embeddings
        pairs = self.accumulator.pairs

        labels = metric.compute_labels(self.accumulator.labels, pairs, self.accumulator.subgroups)

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
        else:
            ref_embeddings = embeddings[sample_indices]

        distance_matrix = metric.distance.distance_matrix(ref_embeddings, embeddings)
        return labels.float(), distance_matrix
