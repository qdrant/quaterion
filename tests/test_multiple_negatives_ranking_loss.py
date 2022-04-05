import torch

from quaterion.distances import Distance
from quaterion.loss import MultipleNegativesRankingLoss


class TestMNRLoss:
    embeddings = torch.Tensor(
        [
            [0.0, -1.0, 0.5],
            [0.1, 2.0, 0.5],
            [0.0, 0.3, 0.2],
            [1.0, 0.0, 0.9],
            [1.2, -1.2, 0.01],
            [-0.7, 0.0, 1.5],
        ]
    )

    pairs = torch.LongTensor([[0, 3], [1, 4], [2, 5]])

    def test_default_args(self):
        loss = MultipleNegativesRankingLoss()

        loss_res = loss.forward(self.embeddings, self.pairs, torch.Tensor(), torch.Tensor())

        assert loss_res.shape == torch.Size([])

    def test_dot_product(self):
        loss = MultipleNegativesRankingLoss(scale=1, distance_metric_name=Distance.DOT_PRODUCT)

        loss_res = loss.forward(self.embeddings, self.pairs, torch.Tensor(), torch.Tensor())

        assert loss_res.shape == torch.Size([])

    def test_symmetric(self):
        loss = MultipleNegativesRankingLoss(symmetric=True)

        loss_res = loss.forward(self.embeddings, self.pairs, torch.Tensor(), torch.Tensor())

        assert loss_res.shape == torch.Size([])
