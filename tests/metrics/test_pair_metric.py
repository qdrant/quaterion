import pytest
import torch
from .params import retrieval_precision_params, retrieval_reciprocal_rank_params
from quaterion.eval.pair_metric import RetrievalPrecision, RetrievalReciprocalRank


def test_retrieval_precision(retrieval_precision_params):
    encoder, distance_fn, k, data, exp_metric = retrieval_precision_params
    metric = RetrievalPrecision(encoder, distance_fn, k)
    for batch in data:
        metric.update(batch)
    assert torch.allclose(metric.compute(), exp_metric)


def test_retrieval_reciprocal_rank(retrieval_reciprocal_rank_params):
    encoder, distance_fn, data, exp_metric = retrieval_reciprocal_rank_params
    metric = RetrievalReciprocalRank(encoder, distance_fn)
    for batch in data:
        metric.update(batch)
    assert torch.allclose(metric.compute(), exp_metric)
