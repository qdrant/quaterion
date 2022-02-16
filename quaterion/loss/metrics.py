import torch

from torch import Tensor, pairwise_distance, cosine_similarity
from torch.nn import functional as F


class SiameseDistanceMetric:
    """The metric for the contrastive loss."""

    @staticmethod
    def euclidean(x: Tensor, y: Tensor, matrix=False) -> Tensor:
        """Compute euclidean distance

        Args:
            x: shape: (batch_size, ...)
            y: shape: (batch_size, ...)
            matrix: flat to calculate distance matrix (all to all)
        Returns:
            Tensor: shape (batch_size, 1)
        """
        if not matrix:
            return pairwise_distance(x, y, p=2)
        raise NotImplementedError()

    @staticmethod
    def manhattan(x: Tensor, y: Tensor, matrix=False) -> Tensor:
        """Compute manhattan distance

        Args:
            x: shape: (batch_size, ...)
            y: shape: (batch_size, ...)
            matrix: flat to calculate distance matrix (all to all)
        Returns:
            Tensor: shape (batch_size, 1)
        """
        if not matrix:
            return pairwise_distance(x, y, p=1)
        raise NotImplementedError()

    @staticmethod
    def cosine_distance(x: Tensor, y: Tensor, matrix=False) -> Tensor:
        """Compute cosine distance

        Args:
            x: shape: (batch_size, ...)
            y: shape: (batch_size, ...)
            matrix: flat to calculate distance matrix (all to all)
        Returns:
            Tensor: shape (batch_size, 1)
        """

        if not matrix:
            return 1 - cosine_similarity(x, y)

        x_norm = F.normalize(x, p=2, dim=1)
        y_norm = F.normalize(y, p=2, dim=1).transpose(0, 1)
        return 1 - torch.mm(x_norm, y_norm)

    @staticmethod
    def dot_product_distance(x: Tensor, y: Tensor, matrix=False) -> Tensor:
        """Compute dot product distance

        Dot product distance may have unexpected effects. Make sure you entirely
        understand how it works itself, and with chosen loss function especially.

        Args:
            x: shape: (batch_size, ...)
            y: shape: (batch_size, ...)
            matrix: flat to calculate distance matrix (all to all)
        Returns:
            Tensor: shape (batch_size, 1)
        """
        return (
            -torch.einsum("id,jd->ij", x, y)
            if matrix
            else -torch.einsum("id,id->i", x, y)
        )
