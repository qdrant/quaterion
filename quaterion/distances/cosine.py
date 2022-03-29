import torch
from quaterion.distances.base_distance import base_distance
from torch import Tensor
import torch.nn.functional as F

class Cosine(BaseDistance):
    """Compute cosine similarities (and its interpretation as distances).

    Note:
        The output range of this metric is `0, -> 1`.
    """

    def similarity(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.cosine_similarity(x, y)

    def distance(self, x: Tensor, y: Tensor) -> Tensor:
        return 1 - self.distance(x, y)

    def similarity_matrix(self, x: Tensor, y: Tensor = None) -> Tensor:
        x_norm = F.normalize(x, p=2, dim=1)
        if y is None:
            y_norm = x_norm.transpose(0, 1)
        else:
            y_norm = F.normalize(x, p=2, dim=1).transpose(0, 1)

        return (torch.mm(x_norm, y_norm) + 1) / 2

    def distance_matrix(self, x: Tensor, y: Tensor = None) -> Tensor:
        return 1 - self.similarity_matrix(x, y)
