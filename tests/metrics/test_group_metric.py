from .params import retrieval_r_precision_params
from quaterion.eval.group_metric import RetrievalRPrecision


def test_retrieval_r_precision(retrieval_r_precision_params):
    distance_fn, data, exp_metric = retrieval_r_precision_params

    metric = RetrievalRPrecision(distance_fn)

    for batch in data:
        embeddings, groups = batch
        metric.update(embeddings, groups)
    assert metric.compute() == exp_metric
