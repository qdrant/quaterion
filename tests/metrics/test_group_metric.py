import torch

from quaterion.eval.group_metric import RetrievalRPrecision


def dummy_distance_fn(x, y, matrix=True):
    return x


def test_retrieval_r_precision():
    # region dummy tests

    # distances aren't actually calculated, tensor already contains assumed distances
    # avoids distance calculation and test only metric logic

    # region single batch
    embeddings = torch.Tensor(
        [
            [0.0, 0.3, 0.9, 0.6],
            [0.3, 0.0, 0.4, 0.15],
            [0.9, 0.4, 0.0, 0.8],
            [0.6, 0.15, 0.8, 0.0],
        ]
    )
    groups = torch.LongTensor([1, 2, 1, 2])
    exp_metric = torch.Tensor([0.5])
    metric = RetrievalRPrecision(dummy_distance_fn)
    metric.update(embeddings, groups)
    assert metric.compute() == exp_metric
    # endregion single batch

    # region multiple batches
    first_batch = (
        torch.Tensor(
            [
                [0.0, 0.3, 0.9, 0.6],
                [0.3, 0.0, 0.4, 0.15],
            ]
        ),  # embeddings
        torch.LongTensor([1, 2]),  # groups
    )
    second_batch = (
        torch.Tensor([[0.9, 0.4, 0.0, 0.8], [0.6, 0.15, 0.8, 0.0]]),  # embeddings
        torch.LongTensor([1, 2]),  # groups
    )

    exp_metric = torch.Tensor([0.5])
    metric = RetrievalRPrecision(dummy_distance_fn)

    for batch in (first_batch, second_batch):
        embeddings, groups = batch
        metric.update(embeddings, groups)
    assert metric.compute() == exp_metric
    # endregion multiple batches
    # endregion dummy test
