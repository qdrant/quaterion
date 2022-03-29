import pytest
import torch
from .params import retrieval_precision_params, retrieval_reciprocal_rank_params
from quaterion.eval.pair_metric import RetrievalPrecision, RetrievalReciprocalRank


def test_retrieval_precision(retrieval_precision_params):
    distance_fn, k, data, exp_metric = retrieval_precision_params
    metric = RetrievalPrecision(distance_fn, k)
    for batch in data:
        embeddings, pairs, labels, subgroups = batch
        metric.update(embeddings, pairs, labels, subgroups)
    assert torch.allclose(metric.compute(), exp_metric)


def test_retrieval_reciprocal_rank(retrieval_reciprocal_rank_params):
    distance_fn, data, exp_metric = retrieval_reciprocal_rank_params
    metric = RetrievalReciprocalRank(distance_fn)
    for batch in data:
        embeddings, pairs, labels, subgroups = batch
        metric.update(embeddings, pairs, labels, subgroups)
    assert torch.allclose(metric.compute(), exp_metric)
