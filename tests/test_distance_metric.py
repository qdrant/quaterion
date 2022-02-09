import torch

from quaterion.loss import SiameseDistanceMetric


def test_distance_metric():
    x = torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )

    res = SiameseDistanceMetric.cosine_distance(x, x, matrix=True)

    print("\n", res)
