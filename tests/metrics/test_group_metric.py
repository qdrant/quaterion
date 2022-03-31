import torch

from quaterion.eval.group_metric import RetrievalRPrecision
from quaterion.loss import SiameseDistanceMetric


def sample_embeddings(mean, std, mean_coef, num_objects, embedding_dim):
    """Generates embeddings from different distributions for tests

    Since metrics might be hard to implement and to test, it might be useful to test them with
    random sampling.

    Here 2 sets of embeddings being generated.
    All embeddings in the first set sampled from one distribution.
    Embeddings in the second set sampled from different distributions.
    """
    size = num_objects, embedding_dim
    x = torch.normal(mean=mean, std=std, size=size)
    y = torch.normal(mean=mean, std=std, size=size)
    z = torch.normal(mean=mean * mean_coef, std=std, size=size)

    same_dist_embeddings = torch.cat([x, y])
    diff_dist_embeddings = torch.cat([x, z])
    return same_dist_embeddings, diff_dist_embeddings


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

    # region ideal case
    embeddings = torch.Tensor([[1, 1], [8, 8], [2, 2], [6, 6]])
    # distance matrix
    # [
    #     [0., 14., 2., 10.],
    #     [14., 0., 12., 4.],
    #     [2., 12., 0., 8.],
    #     [10., 4., 8., 0.]
    # ]
    groups = torch.LongTensor([1, 2, 1, 2])
    exp_metric = torch.Tensor([1.0])
    metric = RetrievalRPrecision(SiameseDistanceMetric.manhattan)
    metric.update(embeddings, groups)
    assert metric.compute() == exp_metric
    # endregion ideal case

    # region worst case
    embeddings = torch.Tensor([[1, 1], [2, 2], [8, 8], [6, 6]])
    # distance matrix
    # [
    #      [0., 2., 14., 10.],
    #      [2., 0., 12., 8.],
    #      [14., 12., 0., 4.],
    #      [10., 8., 4., 0.]
    # ]
    groups = torch.LongTensor([1, 2, 1, 2])
    exp_metric = torch.Tensor([0.0])
    metric = RetrievalRPrecision(SiameseDistanceMetric.manhattan)
    metric.update(embeddings, groups)
    assert metric.compute() == exp_metric
    # endregion worst case

    # region random sampling
    num_objects = 100
    groups = torch.LongTensor([1] * num_objects + [2] * num_objects)

    same_dist_embeddings, diff_dist_embeddings = sample_embeddings(
        mean=1, std=1, mean_coef=2, num_objects=num_objects, embedding_dim=10,
    )
    same_dist_metric = RetrievalRPrecision(SiameseDistanceMetric.manhattan)
    same_dist_metric.update(same_dist_embeddings, groups)

    diff_dist_metric = RetrievalRPrecision(SiameseDistanceMetric.manhattan)
    diff_dist_metric.update(diff_dist_embeddings, groups)
    assert same_dist_metric.compute() <= diff_dist_metric.compute()
    # endregion random sampling
