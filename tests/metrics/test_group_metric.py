import torch

from .params import retrieval_r_precision_params
from quaterion.eval.group_metric import RetrievalRPrecision


def test_retrieval_r_precision(retrieval_r_precision_params):
    encoder, distance_fn, data, exp_metric = retrieval_r_precision_params

    metric = RetrievalRPrecision(encoder, distance_fn)

    for batch in data:
        metric.update(batch)
    assert metric.compute() == exp_metric

