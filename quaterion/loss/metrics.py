import torch

from torch import Tensor, pairwise_distance, cosine_similarity


class SiameseDistanceMetric:
    """The metric for the contrastive loss."""

    @staticmethod
    def euclidean(x: Tensor, y: Tensor) -> Tensor:
        """Compute euclidean distance

        Args:
            x: shape: (batch_size, ...)
            y: shape: (batch_size, ...)

        Returns:
            Tensor: shape (batch_size, 1)
        """
        return pairwise_distance(x, y, p=2)

    @staticmethod
    def manhattan(x: Tensor, y: Tensor) -> Tensor:
        """Compute manhattan distance

        Args:
            x: shape: (batch_size, ...)
            y: shape: (batch_size, ...)

        Returns:
            Tensor: shape (batch_size, 1)
        """
        return pairwise_distance(x, y, p=1)

    @staticmethod
    def cosine_distance(x: Tensor, y: Tensor) -> Tensor:
        """Compute cosine distance

        Args:
            x: shape: (batch_size, ...)
            y: shape: (batch_size, ...)

        Returns:
            Tensor: shape (batch_size, 1)
        """

        return 1 - cosine_similarity(x, y)

    @staticmethod
    def dot_product_distance(x: Tensor, y: Tensor) -> Tensor:
        """Compute dot product distance

        Dot product distance may have unexpected effects. Make sure you entirely
        understand how it works itself, and with chosen loss function especially.

        Args:
            x: shape: (batch_size, ...)
            y: shape: (batch_size, ...)

        Returns:
            Tensor: shape (batch_size, 1)
        """
        return -torch.einsum("id,jd->j", x, y)
