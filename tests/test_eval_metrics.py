import pytest

import torch

from quaterion.eval.metrics import (
    retrieval_reciprocal_rank_2d,
    retrieval_precision_2d,
)


@pytest.mark.parametrize(
    "rrk_preds, rrk_targets, expected",
    [
        (
            torch.Tensor([[0.2, 0.5, 0.3], [0.1, 0.1, 0.8]]),
            torch.Tensor([[0, 0, 1], [0, 0, 1]]),
            torch.Tensor([0.5, 1]),
        ),
    ],
)
def test_retrieval_reciprocal_rank_2d(rrk_preds, rrk_targets, expected):
    assert torch.equal(retrieval_reciprocal_rank_2d(rrk_preds, rrk_targets), expected)


@pytest.mark.parametrize(
    "rp_preds, rp_targets, expected",
    [
        (
            torch.Tensor([[0.2, 0.5, 0.3], [0.1, 0.1, 0.8]]),
            torch.Tensor([[0, 0, 1], [0, 0, 1]]),
            torch.Tensor([0.0, 1.0]),
        ),
    ],
)
def test_retrieval_reciprocal_precision_2d(rp_preds, rp_targets, expected):
    retrieval_precision_2d(rp_preds, rp_targets)
