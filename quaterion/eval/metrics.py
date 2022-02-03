import torch


def retrieval_reciprocal_rank_2d(preds, target):
    indices = torch.argsort(preds, dim=1, descending=True)
    target = target.gather(1, indices)
    position = torch.nonzero(target)
    res = 1.0 / (position[:, 1] + 1.0)
    return res


def retrieval_precision_2d(preds, target):
    relevant = target.gather(1, preds.topk(1, dim=1)[1]).float()
    return relevant
