import pytest
import torch

from quaterion.distances import Distance
from quaterion.eval.pair import RetrievalPrecision, RetrievalReciprocalRank


def sample_embeddings(mean, std, mean_coef, num_of_pairs, embedding_dim):
    """Generates embeddings from different distributions for tests

    Since metrics might be hard to implement and to test, it might be useful to test them with
    random sampling.

    Here 2 sets of embeddings being generated.
    All embeddings in the first set sampled from one distribution.
    Embeddings in the second set sampled from different distributions. Each distribution samples
    only a pair of embeddings.
    """
    shape = (
        num_of_pairs,
        embedding_dim,
    )  # NOTE: total number of embeddings is 2 * num_of_pairs because embeddings being calculated
    # for EACH object in a pair
    dist_mean = [
        i * mean_coef for i in range(shape[0])
    ]  # means for distributions to sample meaningful pairs

    x = torch.normal(mean=mean, std=std, size=shape)
    y = torch.normal(mean=mean, std=std, size=shape)
    same_dist_embeddings = torch.cat([x, y])

    dist_x = torch.Tensor()
    dist_y = torch.Tensor()
    # x are the first objects, y are the second ones
    # iteratively sample a pair of objects from each distribution
    for m in dist_mean:
        dist_x = torch.cat([dist_x, torch.normal(mean=m, std=std, size=(1, shape[1]))])
        dist_y = torch.cat([dist_y, torch.normal(mean=m, std=std, size=(1, shape[1]))])

    diff_dist_embeddings = torch.cat([dist_x, dist_y])
    return same_dist_embeddings, diff_dist_embeddings


def test_retrieval_precision():
    # region single batch
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
    subgroups = torch.Tensor([0, 0, 0, 0])

    exp_metric = torch.Tensor([1, 1, 0, 1])
    metric = RetrievalPrecision(k, Distance.MANHATTAN, reduce_func=None)
    assert torch.allclose(metric.compute(embeddings, labels, pairs, subgroups), exp_metric)
    # endregion k = 1

    # region k = 2
    k = 2
    exp_metric = torch.Tensor([0.5, 0.5, 0.5, 0.5])
    metric = RetrievalPrecision(
        k=k, distance_metric_name=Distance.MANHATTAN, reduce_func=None
    )
    assert torch.allclose(metric.compute(embeddings=embeddings, pairs=pairs, labels=labels, subgroups=subgroups), exp_metric)
    # endregion k = 2

    # region k = 1, ideal case
    k = 1
    embeddings = torch.Tensor([[1, 1], [8, 8], [2, 2], [6, 6]])
    # distance matrix is:
    # [
    #     [0., 14., 2., 10.],
    #     [14., 0., 12., 4.],
    #     [2., 12., 0., 8.],
    #     [10., 4., 8., 0.]
    # ]
    num_of_pairs = embeddings.shape[0] // 2
    pairs = torch.LongTensor([[i, i + num_of_pairs] for i in range(num_of_pairs)])
    labels = torch.Tensor([1.0, 1.0])
    subgroups = torch.Tensor([0, 0, 0, 0])

    exp_metric = torch.Tensor([1, 1, 1, 1])
    metric = RetrievalPrecision(k, Distance.MANHATTAN, reduce_func=None)
    assert torch.allclose(metric.compute(embeddings, labels, pairs, subgroups), exp_metric)
    # endregion k = 1, ideal case

    # region k = 1, worst case
    k = 1
    embeddings = torch.Tensor([[1, 1], [2, 2], [8, 8], [6, 6]])
    # distance matrix
    # [
    #     [0., 14., 2., 10.],
    #     [14., 0., 12., 4.],
    #     [2., 12., 0., 8.],
    #     [10., 4., 8., 0.]
    # ]
    num_of_pairs = embeddings.shape[0] // 2
    pairs = torch.LongTensor([[i, i + num_of_pairs] for i in range(num_of_pairs)])
    labels = torch.Tensor([1.0, 1.0])
    subgroups = torch.Tensor([0, 0, 0, 0])

    exp_metric = torch.Tensor([0, 0, 0, 0])
    metric = RetrievalPrecision(k, Distance.MANHATTAN, reduce_func=None)
    assert torch.allclose(metric.compute(embeddings, labels, pairs, subgroups), exp_metric)
    # endregion k = 1, worst case

    # region random sampling
    num_of_pairs = 100
    pairs = torch.LongTensor([[i, i + num_of_pairs] for i in range(num_of_pairs)])
    labels = torch.Tensor([1.0] * num_of_pairs)
    subgroups = torch.Tensor(
        [0] * num_of_pairs * 2
    )  # subgroup is required for each object

    same_dist_embeddings, diff_dist_embeddings = sample_embeddings(
        mean=1, std=1, mean_coef=1.1, num_of_pairs=num_of_pairs, embedding_dim=10
    )
    same_dist_metric = RetrievalPrecision(k=1, distance_metric_name=Distance.MANHATTAN)
    same_dist_metric.update(same_dist_embeddings, labels, pairs, subgroups)

    diff_dist_metric = RetrievalPrecision(k=1, distance_metric_name=Distance.MANHATTAN)
    diff_dist_metric.update(diff_dist_embeddings, labels, pairs, subgroups)
    assert same_dist_metric.evaluate() <= diff_dist_metric.evaluate()
    # endregion random sampling
    # endregion single batch


def test_retrieval_reciprocal_rank():
    # region single batch
    # region common case
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
    subgroups = torch.Tensor([0, 0, 0, 0])
    exp_metric = torch.Tensor([1, 1, 0.5, 1])

    metric = RetrievalReciprocalRank(Distance.MANHATTAN, reduce_func=None)
    assert torch.allclose(metric.compute(embeddings=embeddings, pairs=pairs, labels=labels, subgroups=subgroups), exp_metric)
    # endregion common case

    # region ideal case
    embeddings = torch.Tensor([[1, 1], [8, 8], [4, 4], [9, 9]])
    # distance matrix is:
    # [
    #     [0., 14., 6., 16.],
    #     [14., 0., 8., 2.],
    #     [6., 8., 0., 10.],
    #     [16., 2., 10., 0.]
    # ]
    num_of_pairs = embeddings.shape[0] // 2
    pairs = torch.LongTensor([[i, i + num_of_pairs] for i in range(num_of_pairs)])
    labels = torch.Tensor([1.0, 1.0])
    subgroups = torch.Tensor([0, 0, 0, 0])
    exp_metric = torch.Tensor([1, 1, 1, 1])

    metric = RetrievalReciprocalRank(Distance.MANHATTAN, reduce_func=None)
    assert torch.allclose(metric.compute(embeddings, labels, pairs, subgroups), exp_metric)
    # endregion ideal case

    # region worst case
    embeddings = torch.Tensor([[3, -1], [2, -3], [0, -3], [0, -1]])
    # distance matrix is:
    # [
    #     [0., 3., 5., 3.],
    #     [3., 0., 2., 4.],
    #     [5., 2., 0., 2.],
    #     [3., 4., 2., 0.]
    # ]
    num_of_pairs = embeddings.shape[0] // 2
    pairs = torch.LongTensor([[i, i + num_of_pairs] for i in range(num_of_pairs)])
    labels = torch.Tensor([1.0, 1.0])
    subgroups = torch.Tensor([0, 0, 0, 0])
    exp_metric = torch.Tensor([1 / 3, 1 / 3, 1 / 3, 1 / 3])

    metric = RetrievalReciprocalRank(Distance.MANHATTAN)
    assert torch.allclose(metric.compute(embeddings, labels, pairs, subgroups), exp_metric)
    # endregion worst case

    # region random sampling
    num_of_pairs = 100
    pairs = torch.LongTensor([[i, i + num_of_pairs] for i in range(num_of_pairs)])
    labels = torch.Tensor([1.0] * num_of_pairs)
    subgroups = torch.Tensor(
        [0] * num_of_pairs * 2
    )  # subgroup is required for each object

    same_dist_embeddings, diff_dist_embeddings = sample_embeddings(
        mean=1, std=1, mean_coef=1.1, num_of_pairs=num_of_pairs, embedding_dim=10
    )
    same_dist_metric = RetrievalReciprocalRank(Distance.MANHATTAN)
    same_dist_metric.update(same_dist_embeddings, labels, pairs, subgroups)

    diff_dist_metric = RetrievalReciprocalRank(Distance.MANHATTAN)
    diff_dist_metric.update(diff_dist_embeddings, labels, pairs, subgroups)
    assert same_dist_metric.evaluate() <= diff_dist_metric.evaluate()
    # endregion random sampling
    # endregion single batch
