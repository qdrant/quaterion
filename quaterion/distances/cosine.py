import torch
import torch.nn.functional as F
from torch import Tensor

from quaterion.distances.base_distance import BaseDistance


class Cosine(BaseDistance):
    """Compute cosine similarities (and its interpretation as distances).

    Note:
        The output range of this metric is `0 -> 1`.
    """

    @staticmethod
    def similarity(x: Tensor, y: Tensor) -> Tensor:
        return torch.cosine_similarity(x, y)

    @staticmethod
    def distance(x: Tensor, y: Tensor) -> Tensor:
        return 1 - Cosine.similarity(x, y)

    @staticmethod
    def similarity_matrix(x: Tensor, y: Tensor = None) -> Tensor:
        x_norm = F.normalize(x, p=2, dim=1)
        if y is None:
            y_norm = x_norm.transpose(0, 1)
        else:
            y_norm = F.normalize(x, p=2, dim=1).transpose(0, 1)

        return (torch.mm(x_norm, y_norm) + 1) / 2

    @staticmethod
    def distance_matrix(x: Tensor, y: Tensor = None) -> Tensor:
        return 1 - Cosine.similarity_matrix(x, y)
