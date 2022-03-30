import pytest
import torch

from quaterion.loss.metrics import SiameseDistanceMetric
from quaterion.eval.pair_metric import RetrievalPrecision, RetrievalReciprocalRank


def test_retrieval_precision():
    # region k = 1
    k = 1
    embeddings = torch.Tensor([[1, 1], [8, 8], [4, 4], [6, 7]])
    # distance matrix is:
    # [
    #      [0., 14., 6., 11.],
    #      [14., 0., 8., 3.],
    #      [6., 8., 0., 5.],
    #      [11., 3., 5., 0.]
    # ]
    # nearest pairs (indices) for each row: (0, 2), (1, 3), (2, 3), (3, 1)
    # one mismatch - (2, 3), correct pair is (2, 0)
    num_of_pairs = embeddings.shape[0] // 2
    pairs = torch.LongTensor([[i, i + num_of_pairs] for i in range(num_of_pairs)])
    labels = torch.Tensor([1.0, 1.0])
    subgroups = torch.Tensor([1, 2] * 2)

    exp_metric = torch.Tensor([1, 1, 0, 1])
    metric = RetrievalPrecision(SiameseDistanceMetric.manhattan, k)
    metric.update(embeddings, pairs, labels, subgroups)
    assert torch.allclose(metric.compute(), exp_metric)
    # endregion k = 1

    # region k = 2
    k = 2
    exp_metric = torch.Tensor([0.5, 0.5, 0.5, 0.5])
    metric = RetrievalPrecision(SiameseDistanceMetric.manhattan, k)

    metric.update(embeddings, pairs, labels, subgroups)
    assert torch.allclose(metric.compute(), exp_metric)
    # endregion k = 2


def test_retrieval_reciprocal_rank():
    # region single batch
    embeddings = torch.Tensor([[1, 1], [8, 8], [4, 4], [6, 7]])
    # distance matrix is:
    # [
    #      [0., 14., 6., 11.],
    #      [14., 0., 8., 3.],
    #      [6., 8., 0., 5.],
    #      [11., 3., 5., 0.]
    # ]
    # nearest pairs (indices) for each row: (0, 2), (1, 3), (2, 3), (3, 1)
    # one mismatch - (2, 3), correct pair is (2, 0). Correct pair will be achieved at the second
    # lookup in this row.

    num_of_pairs = embeddings.shape[0] // 2
    pairs = torch.LongTensor([[i, i + num_of_pairs] for i in range(num_of_pairs)])
    labels = torch.Tensor([1.0, 1.0])
    subgroups = torch.Tensor([1, 2] * 2)
    exp_metric = torch.Tensor([1, 1, 0.5, 1])

    metric = RetrievalReciprocalRank(SiameseDistanceMetric.manhattan)
    metric.update(embeddings, pairs, labels, subgroups)
    assert torch.allclose(metric.compute(), exp_metric)
    # endregion single batch
