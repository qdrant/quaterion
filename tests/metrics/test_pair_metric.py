import pytest
import torch

from quaterion.eval.pair_metric import RetrievalPrecision, RetrievalReciprocalRank


def dummy_distance_fn(x, y, matrix=True):
    return x


def test_retrieval_precision():
    # region dummy tests

    # distances aren't actually calculated, embeddings already contains assumed distances
    # avoids distance calculation and test only metric logic

    # region k = 1
    k = 1
    embeddings = torch.Tensor(
        [[1, 0.3, 0.2, 0.5], [0, 1, 0.8, 0.6], [0.5, 0.7, 1, 0.8], [1, 0.2, 0.5, 1]]
    )
    num_of_pairs = embeddings.shape[0] // 2
    pairs = torch.LongTensor([[i, i + num_of_pairs] for i in range(num_of_pairs)])
    labels = torch.Tensor([1.0, 1.0])
    subgroups = torch.Tensor([1, 2] * 2)

    exp_metric = torch.Tensor([1, 0, 1, 1])

    metric = RetrievalPrecision(dummy_distance_fn, k)
    metric.update(embeddings, pairs, labels, subgroups)
    assert torch.allclose(metric.compute(), exp_metric)
    # endregion k = 1

    # region k = 2
    k = 2
    exp_metric = torch.Tensor([0.5, 0.5, 0.5, 0.5])
    metric = RetrievalPrecision(dummy_distance_fn, k)

    metric.update(embeddings, pairs, labels, subgroups)
    assert torch.allclose(metric.compute(), exp_metric)
    # endregion k = 2
    # endregion dummy test


def test_retrieval_reciprocal_rank():
    # region dummy tests

    # distances aren't actually calculated, embeddings already contains assumed distances
    # avoids distance calculation and test only metric logic

    # region single batch
    embeddings = torch.Tensor(
        [[1, 0.3, 0.2, 0.5], [0, 1, 0.8, 0.6], [0.5, 0.7, 1, 0.8], [1, 0.2, 0.5, 1]]
    )
    num_of_pairs = embeddings.shape[0] // 2
    pairs = torch.LongTensor([[i, i + num_of_pairs] for i in range(num_of_pairs)])
    labels = torch.Tensor([1.0, 1.0])
    subgroups = torch.Tensor([1, 2] * 2)
    exp_metric = torch.Tensor([1, 0.5, 1, 1])

    metric = RetrievalReciprocalRank(dummy_distance_fn)
    metric.update(embeddings, pairs, labels, subgroups)
    assert torch.allclose(metric.compute(), exp_metric)
    # endregion single batch
    # endregion dummy test
