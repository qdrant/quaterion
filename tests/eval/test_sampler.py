import pytest
import torch

from quaterion.eval.group import RetrievalRPrecision
from quaterion.eval.pair import RetrievalReciprocalRank
from quaterion.eval.samplers.group_sampler import GroupSampler
from quaterion.eval.samplers.pair_sampler import PairSampler
from quaterion_models import MetricModel
from quaterion_models.encoders import Encoder
from quaterion_models.heads import EmptyHead


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
        return 42

    def forward(self, batch):
        pass

    def save(self, output_path: str):
        pass

    @classmethod
    def load(cls, input_path: str) -> Encoder:
        pass


@pytest.mark.skip(reason="not yet updated")
def test_pair_sampler(dummy_model):
    metric = RetrievalReciprocalRank()

    embeddings = torch.Tensor(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15],
            [16, 17, 18],
            [19, 20, 21],
            [22, 23, 24],
        ]
    )
    labels = torch.Tensor([1, 1, 1, 1])
    pairs = torch.LongTensor([[0, 4], [1, 5], [2, 6], [3, 7]])
    subgroups = torch.Tensor([0, 0, 0, 0, 0, 0, 0, 0])

    # full dataset
    metric.update(embeddings, labels, pairs, subgroups)
    pair_sampler = PairSampler()
    metric_labels, distance_matrix = pair_sampler.sample(metric)
    assert distance_matrix.shape == torch.Size(
        (metric.embeddings.shape[0], metric.embeddings.shape[0])
    )
    assert metric_labels.shape == distance_matrix.shape
    metric.reset()

    # part of dataset
    metric.update(embeddings, labels, pairs, subgroups)
    sample_size = 4
    pair_sampler = PairSampler(sample_size=sample_size)
    metric_labels, distance_matrix = pair_sampler.sample(metric)
    assert distance_matrix.shape == torch.Size(
        (sample_size, metric.embeddings.shape[0])
    )
    assert metric_labels.shape == distance_matrix.shape
    metric.reset()

    # # full dataset distinguish
    metric.update(embeddings, labels, pairs, subgroups)
    pair_sampler = PairSampler(distinguish=True)
    metric_labels, distance_matrix = pair_sampler.sample(metric)
    assert distance_matrix.shape == torch.Size(
        (metric.embeddings.shape[0] // 2, metric.embeddings.shape[0] // 2)
    )
    assert metric_labels.shape == distance_matrix.shape
    metric.reset()

    # part of dataset distinguish
    metric.update(embeddings, labels, pairs, subgroups)
    sample_size = 4
    pair_sampler = PairSampler(sample_size=sample_size, distinguish=True)
    metric_labels, distance_matrix = pair_sampler.sample(metric)
    assert distance_matrix.shape == torch.Size(
        (sample_size, metric.embeddings.shape[0] // 2)
    )
    assert metric_labels.shape == distance_matrix.shape
    metric.reset()


@pytest.mark.skip(reason="not yet updated")
def test_group_sampler(dummy_model):
    metric = RetrievalRPrecision()
    embeddings = torch.Tensor(
        [
            [0, 1, 2],
            [3, 4, 5],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15],
            [16, 17, 18],
        ]
    )
    groups = torch.LongTensor([0, 0, 0, 1, 1, 1])

    # full dataset
    metric.update(embeddings, groups)
    group_sampler = GroupSampler()
    distance_matrix, labels = group_sampler.sample(metric)
    assert distance_matrix.shape == torch.Size(
        (metric.embeddings.shape[0], metric.embeddings.shape[0])
    )
    assert labels.shape == distance_matrix.shape
    metric.reset()

    # part of dataset
    metric.update(embeddings, groups)
    sample_size = 4
    group_sampler = GroupSampler(sample_size=sample_size)
    metric_labels, distance_matrix = group_sampler.sample(metric)
    assert distance_matrix.shape == torch.Size(
        (sample_size, metric.embeddings.shape[0])
    )
    assert metric_labels.shape == distance_matrix.shape
    metric.reset()
