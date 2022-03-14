import torch
from quaterion.loss import SiameseDistanceMetric


class TestDistanceMetric:
    x = torch.tensor(
        [
            [1.0, -1.5, 2.0, -3.0],
            [-1.0, 1.5, -2.0, 3.0],
        ]
    )

    x_dim = x.size()[0]
    expected_cosine_distance = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    expected_manhattan_distance = torch.tensor([[0, 15], [15, 0]])

    def test_cosine_distance_matrix(self):
        res = SiameseDistanceMetric.cosine_distance(self.x, matrix=True)
        assert res.size()[0] == self.x_dim and res.size()[1] == self.x_dim
        assert (res == self.expected_cosine_distance).float().sum() == self.x_dim**2

    def test_manhattan_distance_matrix(self):
        res = SiameseDistanceMetric.manhattan(self.x, matrix=True)
        assert res.size()[0] == self.x_dim and res.size()[1] == self.x_dim
        assert (
            res == self.expected_manhattan_distance
        ).float().sum() == self.x_dim**2
