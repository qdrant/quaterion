import torch

from quaterion.loss import ContrastiveLoss


class TestContrastiveLoss:
    def test_forward(self):
        loss = ContrastiveLoss(size_average=False)

        embeddings = torch.Tensor(
            [
                [0.0, 1.0, 0.5],
                [0.1, 2.0, 0.0],
                [0.0, 0.5, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )

        pairs = torch.LongTensor([[0, 1], [0, 2], [0, 3], [1, 3], [2, 3]])

        labels = torch.Tensor([1, 0, 0, 1, 0])

        loss_res = loss.forward(embeddings=embeddings, pairs=pairs, labels=labels)

        assert loss_res.shape == torch.Size([])
