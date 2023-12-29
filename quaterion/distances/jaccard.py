from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from quaterion.distances.base_distance import BaseDistance


class Jaccard(BaseDistance):
    """Compute Weighted Jaccard distances (and its interpretation as similarities).

    Note:
        The implementation of Weighted Jaccard
        (https://en.wikipedia.org/wiki/Jaccard_index#Weighted_Jaccard_similarity_and_distance)
        supports Tensors with positive float values.
    """

    @staticmethod
    def distance(x: Tensor, y: Tensor) -> Tensor:
        return 1 - Jaccard.similarity(x, y)

    @staticmethod
    def similarity(x: Tensor, y: Tensor) -> Tensor:
        min_sum = torch.minimum(x, y).sum(dim=-1)
        max_sum = torch.maximum(x, y).sum(dim=-1)
        return min_sum / max_sum

    @staticmethod
    def distance_matrix(x: Tensor, y: Optional[Tensor] = None) -> Tensor:
        return 1 - Jaccard.similarity_matrix(x.unsqueeze(1), y.unsqueeze(0))

    @staticmethod
    def similarity_matrix(x: Tensor, y: Optional[Tensor] = None) -> Tensor:
        if y is None:
            y = x
        return Jaccard.similarity(x.unsqueeze(1), y.unsqueeze(0))
