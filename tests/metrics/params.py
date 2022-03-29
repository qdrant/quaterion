import pytest

import torch


def dummy_metric_fn(x, y, matrix=True):
    return x


def _retrieval_r_precision_params():
    params = []

    # region dummy tests

    # distances aren't actually calculated, tensor already contains assumed distances
    # avoids distance calculation and test only metric logic

    # region 1st test
    dummy_batch = (
        torch.Tensor(
            [
                [0.0, 0.3, 0.9, 0.6],
                [0.3, 0.0, 0.4, 0.15],
                [0.9, 0.4, 0.0, 0.8],
                [0.6, 0.15, 0.8, 0.0],
            ]
        ),
        torch.LongTensor([1, 2, 1, 2]),
    )
    params.append((dummy_metric_fn, [dummy_batch], torch.Tensor([0.5])))
    # endregion 1st test

    # region 2nd test
    batches = list()
    batches.append(
        (
            torch.Tensor([[0.0, 0.3, 0.9, 0.6], [0.3, 0.0, 0.4, 0.15],]),
            torch.LongTensor([1, 2]),
        )
    )
    batches.append(
        (
            torch.Tensor([[0.9, 0.4, 0.0, 0.8], [0.6, 0.15, 0.8, 0.0]]),
            torch.LongTensor([1, 2]),
        )
    )
    params.append((dummy_metric_fn, batches, torch.Tensor([0.5])))
    # endregion 2nd test
    # endregion dummy test
    return params


@pytest.fixture(params=_retrieval_r_precision_params())
def retrieval_r_precision_params(request):
    yield request.param


def _retrieval_precision_params():
    params = []
    # region dummy tests

    # distances aren't actually calculated, embeddings already contains assumed distances
    # avoids distance calculation and test only metric logic

    # region 1st test
    embeddings = torch.Tensor(
        [[1, 0.3, 0.2, 0.5], [0, 1, 0.8, 0.6], [0.5, 0.7, 1, 0.8], [1, 0.2, 0.5, 1]]
    )
    num_of_pairs = embeddings.shape[0] // 2
    dummy_batch = (
        embeddings,
        torch.LongTensor([[i, i + num_of_pairs] for i in range(num_of_pairs)]),
        torch.Tensor([1.0, 1.0]),
        torch.Tensor([1, 2] * 2),
    )
    params.append((dummy_metric_fn, 1, [dummy_batch], torch.Tensor([1, 0, 1, 1])))
    # endregion
    # region 2nd test
    params.append(
        (dummy_metric_fn, 2, [dummy_batch], torch.Tensor([0.5, 0.5, 0.5, 0.5]),)
    )
    # endregion
    # endregion
    return params


@pytest.fixture(params=_retrieval_precision_params())
def retrieval_precision_params(request):
    yield request.param


def _retrieval_reciprocal_rank_params():
    params = []
    # region dummy tests

    # distances aren't actually calculated, embeddings already contains assumed distances
    # avoids distance calculation and test only metric logic

    # region 1st test
    embeddings = torch.Tensor(
        [[1, 0.3, 0.2, 0.5], [0, 1, 0.8, 0.6], [0.5, 0.7, 1, 0.8], [1, 0.2, 0.5, 1]]
    )
    num_of_pairs = embeddings.shape[0] // 2
    dummy_batch = (
        embeddings,
        torch.LongTensor(
            [[i, i + num_of_pairs] for i in range(num_of_pairs)]
        ),
        torch.Tensor([1.0, 1.0]),
        torch.Tensor([1, 2] * 2),
    )
    params.append((dummy_metric_fn, [dummy_batch], torch.Tensor([1, 0.5, 1, 1])))
    # endregion
    # endregion
    return params


@pytest.fixture(params=_retrieval_reciprocal_rank_params())
def retrieval_reciprocal_rank_params(request):
    yield request.param
