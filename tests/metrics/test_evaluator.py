import pytest
import torch
from quaterion.distances import Distance

from quaterion.eval.evaluator import Evaluator
from quaterion.eval.group import RetrievalRPrecision


def test_evaluator():

    evaluator = Evaluator(
        RetrievalRPrecision(
            compute_on_step=False, distance_metric_name=Distance.MANHATTAN
        ),
        "RetrievalPrecisionEvaluator",
    )

    embeddings = torch.Tensor([[1, 1, 1, 1], [1, 2, 3, 4]])
    groups = torch.Tensor([1, 1])

    evaluator.update(embeddings, groups=groups)

    assert isinstance(evaluator.metric._embeddings, list) and isinstance(
        evaluator.metric._groups, list
    )

    assert torch.allclose(embeddings, evaluator.metric.embeddings) and torch.allclose(
        groups, evaluator.metric.groups
    )
    assert not evaluator.has_been_reset
    assert evaluator.evaluate() == torch.Tensor([1])

    evaluator.update(
        torch.Tensor([[10, 10, 10, 10], [11, 11, 11, 11]]), groups=torch.Tensor([2, 2])
    )
    assert evaluator.evaluate() == torch.Tensor([1])

    evaluator.reset()
    assert evaluator.has_been_reset
    assert evaluator.metric.embeddings.shape == torch.Size([0])
    assert evaluator.metric.groups.shape == torch.Size([0])

    evaluator = Evaluator(
        RetrievalRPrecision(
            compute_on_step=False,
            distance_metric_name=Distance.MANHATTAN,
        ),
        "RetrievalRPrecisionEvaluator",
        batch_size=2,
    )
    evaluator.update(
        torch.Tensor([[1, 1, 1, 1], [1, 2, 3, 4]]), groups=torch.Tensor([1, 1])
    )
    evaluator.update(
        torch.Tensor([[10, 10, 10, 10], [11, 11, 11, 11]]), groups=torch.Tensor([2, 2])
    )
    assert evaluator.evaluate() == torch.Tensor([1.0])
