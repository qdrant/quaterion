import torch
from quaterion.loss import SiameseDistanceMetric


class TestDistanceMetric:
    x = torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )

    x_dim = x.size()[0]

    def test_cosine_distance_matrix(self):
        res = SiameseDistanceMetric.cosine_distance(self.x, matrix=True)
        assert res.size()[0] == self.x_dim and res.size()[1] == self.x_dim

    def test_manhattan_distance_matrix(self):
        res = SiameseDistanceMetric.manhattan(self.x, matrix=True)
        assert res.size()[0] == self.x_dim and res.size()[1] == self.x_dim
