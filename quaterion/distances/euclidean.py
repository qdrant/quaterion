from typing import Optional

import torch
from torch import Tensor
import torch.nn.functional as F

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

        dot_product = torch.mm(x, y.transpose(0, 1))

        # get L2 norm by diagonal. Shape: (batch_size,)
        square_norm = torch.diagonal(dot_product)
        # calculate distances. Shape: (batch_size, batch_size)
        distance_matrix = (
            square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
        )
        # get rid of negative distances due to calculation errors
        distance_matrix = F.relu(distance_matrix)

        # handle numerical stability to avoid None gradients during backpropogation
        mask = (distance_matrix == 0.0).float()
        distance_matrix = distance_matrix + mask * 1e-16

        distance_matrix = torch.sqrt(distance_matrix)

        # Undo trick for numerical stability
        distance_matrix = distance_matrix * (1.0 - mask)

        return distance_matrix

    @staticmethod
    def similarity_matrix(x: Tensor, y: Optional[Tensor] = None) -> Tensor:
        return 1 / (1 + Euclidean.distance_matrix(x, y))
