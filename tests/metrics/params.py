import pytest
import torch


class DummyEncoder:
    def __call__(self, batch):
        return batch


def dummy_metric_fn(x, y, matrix=True):
    return x


def _retrieval_r_precision_params():
    params = []

    # region dummy tests

    # distances aren't actually calculated, tensor already contains assumed distances
    # avoids distance calculation and test only metric logic

    # region 1st test
    dummy_batch = (
        [1, 2, 3, 4],
        torch.Tensor(
            [
                [0.0, 0.3, 0.9, 0.6],
                [0.3, 0.0, 0.4, 0.15],
                [0.9, 0.4, 0.0, 0.8],
                [0.6, 0.15, 0.8, 0.0],
            ]
        ),
        {"groups": torch.LongTensor([1, 2, 1, 2])},
    )
    params.append((DummyEncoder(), dummy_metric_fn, [dummy_batch], torch.Tensor([0.5])))
    # endregion 1st test

    # region 2nd test
    batches = list()
    batches.append(
        (
            [1, 2,],
            torch.Tensor([[0.0, 0.3, 0.9, 0.6], [0.3, 0.0, 0.4, 0.15],]),
            {"groups": torch.LongTensor([1, 2])},
        )
    )
    batches.append(
        (
            [3, 4],
            torch.Tensor([[0.9, 0.4, 0.0, 0.8], [0.6, 0.15, 0.8, 0.0]]),
            {"groups": torch.LongTensor([1, 2])},
        )
    )
    params.append((DummyEncoder(), dummy_metric_fn, batches, torch.Tensor([0.5])))
    # endregion 2nd test
    # endregion dummy test
    return params


@pytest.fixture(params=_retrieval_r_precision_params())
def retrieval_r_precision_params(request):
    yield request.param


def _retrieval_precision_params():
    params = []
    # region dummy tests
    # region 1st test
    indices = [1, 2]
    dummy_batch = (
        indices,
        torch.Tensor(
            [[1, 0.3, 0.2, 0.5], [0, 1, 0.8, 0.6], [0.5, 0.7, 1, 0.8], [1, 0.2, 0.5, 1]]
        ),
        {
            "pairs": torch.LongTensor(
                [[i, i + len(indices)] for i in range(len(indices))]
            ),
            "labels": torch.Tensor([1.0, 1.0]),
            "subgroups": torch.Tensor([1, 2] * 2),
        },
    )
    params.append(
        (DummyEncoder(), dummy_metric_fn, 1, [dummy_batch], torch.Tensor([1, 0, 1, 1]))
    )
    # endregion
    # region 2nd test
    params.append(
        (DummyEncoder(), dummy_metric_fn, 2, [dummy_batch], torch.Tensor([0.5, 0.5, 0.5, 0.5]))
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
    # region 1st test
    indices = [1, 2]
    dummy_batch = (
        indices,
        torch.Tensor(
            [[1, 0.3, 0.2, 0.5], [0, 1, 0.8, 0.6], [0.5, 0.7, 1, 0.8], [1, 0.2, 0.5, 1]]
        ),
        {
            "pairs": torch.LongTensor(
                [[i, i + len(indices)] for i in range(len(indices))]
            ),
            "labels": torch.Tensor([1.0, 1.0]),
            "subgroups": torch.Tensor([1, 2] * 2),
        },
    )
    params.append(
        (DummyEncoder(), dummy_metric_fn, [dummy_batch], torch.Tensor([1, 0.5, 1, 1]))
    )
    # endregion
    # endregion
    return params


@pytest.fixture(params=_retrieval_reciprocal_rank_params())
def retrieval_reciprocal_rank_params(request):
    yield request.param
