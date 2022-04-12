from typing import Optional

import torch
from torch import Tensor

from quaterion.distances.base_distance import BaseDistance


class Manhattan(BaseDistance):
    """Compute Manhattan distances (and its interpretation as similarities).

    Note:
        Interpretation of Manhattan distances as similarities is based on the trick in the book
        "Collective Intelligence" by Toby Segaran, and it's in the range of `0 -> 1`.
    """

    @staticmethod
    def distance(x: Tensor, y: Tensor) -> Tensor:
        return torch.pairwise_distance(x, y, p=1)

    @staticmethod
    def similarity(x: Tensor, y: Tensor) -> Tensor:
        return 1 / (1 + Manhattan.distance(x, y))

    @staticmethod
    def distance_matrix(x: Tensor, y: Optional[Tensor] = None) -> Tensor:
        if y is None:
            y = x

        # expand dimensions to calculate element-wise differences with broadcasting
        # shape: (batch_size, batch_size, embedding_dim)
        deltas = x.unsqueeze(1) - y.unsqueeze(0)
        abs_deltas = torch.abs(deltas)

        # sum across the last dimension for reduction
        # shape: (batch_size, batch_size)
        distance_matrix = abs_deltas.sum(dim=-1)

        return distance_matrix

    @staticmethod
    def similarity_matrix(x: Tensor, y: Optional[Tensor] = None) -> Tensor:
        return 1 / (1 + Manhattan.distance_matrix(x, y))
