import pytest
import torch
from quaterion_models import MetricModel
from quaterion_models.encoders import Encoder
from quaterion_models.heads import EmptyHead

from quaterion.dataset import SimilarityPairSample, SimilarityGroupSample
from quaterion.distances import Distance
from quaterion.eval.group import RetrievalRPrecision
from quaterion.eval.pair import RetrievalReciprocalRank
from quaterion.eval.samplers.group_sampler import GroupSampler
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


def test_pair_sampler(dummy_model):
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
    metric_labels, distance_matrix = pair_sampler.sample(
        pair_samples, metric, dummy_model
    )
    assert distance_matrix.shape == torch.Size(
        (len(objects_to_encode), len(objects_to_encode))
    )
    assert metric_labels.shape == distance_matrix.shape
    assert metric.raw_compute(distance_matrix, metric_labels).mean().item() == 1.0

    # part of dataset
    sample_size = 4
    pair_sampler = PairSampler(sample_size=sample_size)
    metric_labels, distance_matrix = pair_sampler.sample(
        pair_samples, metric, dummy_model
    )
    assert distance_matrix.shape == torch.Size((sample_size, len(objects_to_encode)))
    assert metric_labels.shape == distance_matrix.shape
    assert metric.raw_compute(distance_matrix, metric_labels).mean().item() == 1.0

    # # full dataset distinguish
    pair_sampler = PairSampler(distinguish=True)
    metric_labels, distance_matrix = pair_sampler.sample(
        pair_samples, metric, dummy_model
    )
    assert distance_matrix.shape == torch.Size(
        (len(objects_to_encode) // 2, len(objects_to_encode) // 2)
    )
    assert metric_labels.shape == distance_matrix.shape
    assert metric.raw_compute(distance_matrix, metric_labels).mean().item() == 1.0

    # part of dataset distinguish
    sample_size = 4
    pair_sampler = PairSampler(sample_size=sample_size, distinguish=True)
    metric_labels, distance_matrix = pair_sampler.sample(
        pair_samples, metric, dummy_model
    )
    assert distance_matrix.shape == torch.Size(
        (sample_size, len(objects_to_encode) // 2)
    )
    assert metric_labels.shape == distance_matrix.shape
    assert metric.raw_compute(distance_matrix, metric_labels).mean().item() == 1.0


def test_group_sampler(dummy_model):
    metric = RetrievalRPrecision()
    embeddings = [
        [0, 1, 2],
        [3, 4, 5],
        [7, 8, 9],
        [10, 11, 12],
        [13, 14, 15],
        [16, 17, 18],
    ]

    groups = torch.LongTensor([0, 0, 0, 1, 1, 1])

    group_samples = []
    for index in range(len(embeddings)):
        group_samples.append(
            SimilarityGroupSample(obj=embeddings[index], group=groups[index],)
        )
    # full dataset
    group_sampler = GroupSampler()
    distance_matrix, metric_labels = group_sampler.sample(
        group_samples, metric, dummy_model
    )
    assert distance_matrix.shape == torch.Size((len(embeddings), len(embeddings)))
    assert metric_labels.shape == distance_matrix.shape

    # part of dataset
    sample_size = 4
    group_sampler = GroupSampler(sample_size=sample_size)
    metric_labels, distance_matrix = group_sampler.sample(
        group_samples, metric, dummy_model
    )
    assert distance_matrix.shape == torch.Size((sample_size, len(embeddings)))
    assert metric_labels.shape == distance_matrix.shape
