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

        pairs = torch.LongTensor([[0, 2], [1, 3]])

        labels = torch.Tensor([1, 0,])

        subgroups = torch.Tensor([42, 13] * 2)

        target = {
            "pairs": pairs,
            "labels": labels,
            "subgroups": subgroups
        }

        loss_res = loss.forward(
            embeddings=embeddings,
            **target
        )

        assert loss_res.shape == torch.Size([])
