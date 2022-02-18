import torch

from torch import Tensor, pairwise_distance, cosine_similarity
from torch.nn import functional as F


class SiameseDistanceMetric:
    """Common distance metrics for losses."""

    @staticmethod
    def euclidean(
        x: Tensor, y: Tensor = None, matrix: bool = False, squared: bool = True
    ) -> Tensor:
        """Compute euclidean distance

        Args:
            x: shape: (batch_size, ...)
            y: shape: (batch_size, ...), optional.
            matrix: if `True` calculate distance matrix over `x`.
            squared: Squared Euclidean distance or not.

        Returns:
            Tensor: shape (batch_size, batch_size) if `matrix` is `True`, (batch_size, 1) otherwise.
        """
        # TODO: Remove `matrix` and rely only on `if y is not None`
        if not matrix:
            return pairwise_distance(x, y, p=2 if squared else 1)

        # Calculate dot product. Shape: (batch_size, batch_size)
        dot_product = torch.mm(x, x.transpose(0, 1))
        # get L2 norm by diagonal. Shape: (batch_size,)
        square_norm = torch.diagonal(dot_product)
        # calculate distances. Shape: (batch_size, batch_size)
        distances = (
            square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
        )
        # get rid of negative distances due to calculation errors
        distances = torch.maximum(distances, torch.tensor(0.0))

        if not squared:
            # handle numerical stability
            mask = (distances == 0.0).float()
            distances = distances + mask * 1e-16

            distances = torch.sqrt(distances)

            # Undo trick for numerical stability
            distances = distances * (1.0 - mask)

        return distances

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
