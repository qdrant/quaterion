import torch
from torch import Tensor

from quaterion.distances.base_distance import BaseDistance


class DotProduct(BaseDistance):
    """Compute dot product similarities (and its interpretation as distances).

    Warnings:
        Interpretation of dot product as distances may have unexpected effects. Make sure that you
        entirely understand how it exactly works, and when combined and with the chosen loss
        function in particular, because those values are negative.
    """

    @staticmethod
    def similarity(x: Tensor, y: Tensor) -> Tensor:
        return torch.einsum("id,id->i", x, y)

    @staticmethod
    def distance(x: Tensor, y: Tensor) -> Tensor:
        # TODO: think of a wiser way of interpreting dot product as distances, which is also compatible with other distance metrics.
        return -DotProduct.similarity(x, y)

    @staticmethod
    def similarity_matrix(x: Tensor, y: Tensor = None) -> Tensor:
        if y is None:
            y = x

        return torch.einsum("id,jd->ij", x, y)

    @staticmethod
    def distance_matrix(x: Tensor, y: Tensor = None) -> Tensor:
        return -DotProduct.similarity_matrix(x, y)
