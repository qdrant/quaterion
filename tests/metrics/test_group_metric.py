import torch

from quaterion.eval.group_metric import RetrievalRPrecision
from quaterion.loss import SiameseDistanceMetric


def test_retrieval_r_precision():
    # region single batch
    embeddings = torch.Tensor([[1, 1], [8, 8], [4, 4], [6, 7]])
    # [
    #     [0., 14., 6., 11.],
    #     [14., 0., 8., 3.],
    #     [6., 8., 0., 5.],
    #     [11., 3., 5., 0.]
    # ]
    # nearest pairs (by indices): (0, 2), (1, 3), (2, 3), (3, 1)
    # one mismatch - (2, 3), correct pair is (2, 0).
    groups = torch.LongTensor([1, 2, 1, 2])
    exp_metric = torch.Tensor([0.75])
    metric = RetrievalRPrecision(SiameseDistanceMetric.manhattan)
    metric.update(embeddings, groups)
    assert metric.compute() == exp_metric
    # endregion single batch

    # region multiple batches
    first_batch = (
        torch.Tensor([[1, 1], [8, 8],]),  # embeddings
        torch.LongTensor([1, 2]),  # groups
    )
    second_batch = (
        torch.Tensor([[4, 4], [6, 7]]),  # embeddings
        torch.LongTensor([1, 2]),  # groups
    )

    exp_metric = torch.Tensor([0.75])
    metric = RetrievalRPrecision(SiameseDistanceMetric.manhattan)

    for batch in (first_batch, second_batch):
        embeddings, groups = batch
        metric.update(embeddings, groups)
    assert metric.compute() == exp_metric
    # endregion multiple batches
