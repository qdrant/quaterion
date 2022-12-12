from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from quaterion.distances.base_distance import BaseDistance


class Euclidean(BaseDistance):
    """Compute Euclidean distances (and its interpretation as similarities).

    Note:
        Interpretation of Euclidean distances as similarities is based on the trick in the book
        "Collective Intelligence" by Toby Segaran, and it's in the range of `0 -> 1`.

    Note:
        The distance matrix computation is based on `a trick explained by Samuel Albanie
        <https://www.robots.ox.ac.uk/~albanie/notes/Euclidean_distance_trick.pdf>`__.
    """

    @staticmethod
    def distance(x: Tensor, y: Tensor) -> Tensor:
        return torch.pairwise_distance(x, y, p=2)

    @staticmethod
    def similarity(x: Tensor, y: Tensor) -> Tensor:
        return 1 / (1 + Euclidean.distance(x, y))

    @staticmethod
    def distance_matrix(x: Tensor, y: Optional[Tensor] = None) -> Tensor:
        if y is None:
            y = x

        return torch.cdist(x, y)

    @staticmethod
    def similarity_matrix(x: Tensor, y: Optional[Tensor] = None) -> Tensor:
        return 1 / (1 + Euclidean.distance_matrix(x, y))
