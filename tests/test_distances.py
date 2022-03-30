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
        "cosine": {
            "distance_matrix": torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
            "similarity_matrix": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        },
        "manhattan": {
            "distance_matrix": torch.tensor([[0.0, 15.0], [15.0, 0.0]]),
            "similarity_matrix": torch.tensor([[1.0, 0.0625], [0.0625, 1.0]]),
        },
        "euclidean": {
            "distance_matrix": torch.tensor([[0.0, 8.0623], [8.0623, 0.0]]),
            "similarity_matrix": torch.tensor([[1.0, 0.1103477], [0.1103477, 1.0]]),
        },
        "dot_product": {
            "similarity_matrix": torch.tensor([[16.25, -16.25], [-16.25, 16.25]]),
            "distance_matrix": torch.tensor([[-16.25, 16.25], [16.25, -16.25]]),
        },
    }

    @pytest.mark.parametrize(
        ("distance_name", "method_name"),
        [
            ("cosine", "distance_matrix"),
            ("cosine", "similarity_matrix"),
            ("manhattan", "distance_matrix"),
            ("manhattan", "similarity_matrix"),
            ("euclidean", "distance_matrix"),
            ("euclidean", "similarity_matrix"),
            ("dot_product", "distance_matrix"),
            ("dot_product", "similarity_matrix"),
        ],
    )
    def test_distances(self, distance_name, method_name):
        dist_obj = Distance.get_by_name(distance_name)
        if "matrix" in method_name:
            if method_name == "distance_matrix":
                res = dist_obj.distance_matrix(self.x)
            else:  # similarity matrix
                res = dist_obj.similarity_matrix(self.x)

            assert torch.allclose(
                res, self.expected[distance_name][method_name], atol=3e-08
            )  # workaround to avoid small numerical errors
