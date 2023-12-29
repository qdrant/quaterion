import torch

from quaterion.loss import CenterLoss


class TestCenterLoss:
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
    groups = torch.LongTensor([1, 2, 0, 0, 2, 1])

    def test_batch_all(self):
        # Initialize the CenterLoss
        loss = CenterLoss(embedding_size=self.embeddings.size()[1], num_groups=3)

        # Calculate the loss
        loss_res = loss.forward(embeddings=self.embeddings, groups=self.groups)

        # Assertions to check the output shape and type
        assert isinstance(
            loss_res, torch.Tensor
        ), "Loss result should be a torch.Tensor"
        assert loss_res.shape == torch.Size(
            []
        ), "Loss result should be a scalar (0-dimension tensor)"
