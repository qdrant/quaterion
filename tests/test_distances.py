import torch
import pytest
from quaterion.distances import Distance


class TestDistances:
    x = torch.tensor(
        [
            [1.0, -1.5, 2.0, -3.0],
            [-1.0, 1.5, -2.0, 3.0],
        ]
    )

    x_dim = x.size()[0]
    expected = {
        "cosine": {"distance_matrix": torch.tensor([[0.0, 1.0], [1.0, 0.0]])},
        "manhattan": {"distance_matrix": torch.tensor([[0, 15], [15, 0]])},
    }

    @pytest.mark.parametrize(
        ("distance_name", "method_name"),
        [("cosine", "distance_matrix"), ("manhattan", "distance_matrix")],
    )
    def test_distances(self, distance_name, method_name):
        dist_obj = Distance.get_by_name(distance_name)
        if "matrix" in method_name:
            if method_name == "distance_matrix":
                res = dist_obj.distance_matrix(self.x)
            else:  # similarity matrix
                res = dist_obj.similarity_matrix(self.x)

            assert (
                res == self.expected[distance_name][method_name]
            ).float().sum() == self.x_dim ** 2
