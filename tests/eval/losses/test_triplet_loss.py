import torch

from quaterion.loss import TripletLoss


class TestTripletLoss:
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

    groups = torch.LongTensor([1, 2, 3, 3, 2, 1])

    def test_batch_all(self):
        loss = TripletLoss(mining="all")

        loss_res = loss.forward(embeddings=self.embeddings, groups=self.groups)

        assert loss_res.shape == torch.Size([])

    def test_batch_hard(self):
        loss = TripletLoss(mining="hard")

        loss_res = loss.forward(embeddings=self.embeddings, groups=self.groups)

        assert loss_res.shape == torch.Size([])

    def test_xbm(self):
        loss = TripletLoss(mining="hard")
        memory_embeddings = torch.rand(size=(100, 3), dtype=torch.float)
        memory_groups = torch.randint(low=1, high=10, size=(100,), dtype=torch.long)
        regular_loss = loss.forward(self.embeddings, self.groups)
        xbm_loss = loss.forward(
            self.embeddings, self.groups, memory_embeddings, memory_groups
        )

        assert regular_loss.shape == xbm_loss.shape and regular_loss != xbm_loss
