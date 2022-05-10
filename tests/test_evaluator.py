import pytest

import torch
from quaterion_models import MetricModel
from quaterion_models.heads import EmptyHead
from quaterion_models.encoders import Encoder

from quaterion.dataset import SimilarityPairSample
from quaterion.distances import Distance
from quaterion.eval.evaluator import Evaluator
from quaterion.eval.pair import RetrievalReciprocalRank
from quaterion.eval.samplers.pair_sampler import PairSampler


@pytest.fixture
def dummy_model():
    encoder = DummyEncoder()
    head = EmptyHead(encoder.embedding_size)
    yield MetricModel(encoder, head)


class DummyEncoder(Encoder):
    @property
    def trainable(self) -> bool:
        return False

    @property
    def embedding_size(self) -> int:
        return 3

    def forward(self, batch):
        return batch

    def save(self, output_path: str):
        pass

    @classmethod
    def load(cls, input_path: str) -> Encoder:
        pass

    def get_collate_fn(self):
        return self.collate_fn

    def collate_fn(self, batch):
        return torch.Tensor(batch)


def test_evaluator(dummy_model):
    metric = RetrievalReciprocalRank(distance_metric_name=Distance.MANHATTAN)

    objects_to_encode = [
        [1, 1, 1],
        [5, 5, 5],
        [8, 8, 8],
        [11, 11, 11],
        [2, 2, 2],
        [6, 6, 6],
        [9, 9, 9],
        [12, 12, 12],
    ]
    labels = [1, 1, 1, 1]
    pairs = [[0, 4], [1, 5], [2, 6], [3, 7]]
    subgroups = [0, 0, 0, 0, 0, 0, 0, 0]

    pair_samples = []
    for pair in pairs:
        obj_a_index = pair[0]
        obj_b_index = pair[1]
        pair_samples.append(
            SimilarityPairSample(
                obj_a=objects_to_encode[obj_a_index],
                obj_b=objects_to_encode[obj_b_index],
                score=labels[obj_a_index],
                subgroup=subgroups[obj_a_index],
            )
        )

    # full dataset
    pair_sampler = PairSampler()
    evaluator = Evaluator(metrics={"rrk": metric}, sampler=pair_sampler)
    assert evaluator.evaluate(pair_samples, dummy_model) == {"rrk": 1.0}

    # part of dataset
    sample_size = 4
    pair_sampler = PairSampler(sample_size=sample_size)
    evaluator = Evaluator(metrics={"rrk": metric}, sampler=pair_sampler)
    assert evaluator.evaluate(pair_samples, dummy_model) == {"rrk": 1.0}

    # # full dataset distinguish
    pair_sampler = PairSampler(distinguish=True)
    evaluator = Evaluator(metrics={"rrk": metric}, sampler=pair_sampler)
    assert evaluator.evaluate(pair_samples, dummy_model) == {"rrk": 1.0}

    # part of dataset distinguish
    sample_size = 4
    pair_sampler = PairSampler(sample_size=sample_size, distinguish=True)
    evaluator = Evaluator(metrics={"rrk": metric}, sampler=pair_sampler)
    assert evaluator.evaluate(pair_samples, dummy_model) == {"rrk": 1.0}

