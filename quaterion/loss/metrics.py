import torch

from torch import Tensor
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
            y: shape: (batch_size, ...), optional
            matrix: if `True` calculate a distance matrix between `x` and `y` (all-to-all).
                if `y` it is `None`, it assigns `x` to `y`.
            squared: Squared Euclidean distance or not.

        Returns:
            Tensor: shape (batch_size, batch_size) if `matrix` is `True`, (batch_size,) otherwise.
        """
        if not matrix:
            if y is None:
                raise ValueError("y cannot be None while matrix is False")

            distances = torch.pairwise_distance(x, y, p=2)
        else:

            # Calculate dot product. Shape: (batch_size, batch_size)
            if y is None:
                y = x

            dot_product = torch.mm(x, y.transpose(0, 1))

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
    def manhattan(x: Tensor, y: Tensor = None, matrix: bool = False) -> Tensor:
        """Compute manhattan distance

        Args:
            x: shape: (batch_size, ...)
            y: shape: (batch_size, ...), optional
            matrix: if `True` calculate a distance matrix between `x` and `y` (all-to-all).
                if `y` it is `None`, it assigns `x` to `y`.

        Returns:
            Tensor: shape (batch_size, batch_size) if `matrix` is `True`, (batch_size,) otherwise.
        """
        if not matrix:
            if y is None:
                raise ValueError("y cannot be None while matrix is False")

            distances = torch.pairwise_distance(x, y, p=1)

        else:

            if y is None:
                y = x

            # expand dimensions to calculate element-wise diffrences with broadcasting
            # shape: (batch_size, batch_size, vector_dim)
            deltas = x.unsqueeze(1) - y.unsqueeze(0)
            abs_deltas = torch.abs(deltas)

            # sum across the last dimension for reduction
            # shape: (batch_size, batch_size)
            distances = abs_deltas.sum(dim=-1)

        return distances

    @staticmethod
    def cosine_distance(x: Tensor, y: Tensor = None, matrix=False) -> Tensor:
        """Compute cosine distance

        Args:
            x: shape: (batch_size, ...)
            y: shape: (batch_size, ...), optional
            matrix: if `True` calculate a distance matrix between `x` and `y` (all-to-all).
                If `y` is `None`, it assigns `x` to `y`.
        Returns:
            Tensor: shape (batch_size, batch_size) if `matrix` is `True`, (batch_size,) otherwise.
        """

        if not matrix:
            if y is None:
                raise ValueError("y cannot be None while matrix is False")

            return 1 - torch.cosine_similarity(x, y)

        x_norm = F.normalize(x, p=2, dim=1)
        if y is None:
            y_norm = x_norm.transpose(0, 1)
        else:
            y_norm = F.normalize(y, p=2, dim=1).transpose(0, 1)

        # actual interval of cosine similarity is (-1, 1)
        # we need to normalize it to the interval (0, 1) before converting to a distance value
        return 1 - (torch.mm(x_norm, y_norm) + 1) / 2

    @staticmethod
    def dot_product_distance(x: Tensor, y: Tensor, matrix=False) -> Tensor:
        """Compute dot product distance

        Warnings:
            Dot product distance may have unexpected effects. Make sure you entirely understand how
            it works itself, and with chosen loss function especially.

        Args:
            x: shape: (batch_size, ...)
            y: shape: (batch_size, ...)
            matrix: flat to calculate distance matrix (all to all)
        Returns:
            Tensor: shape (batch_size, batch_size) if `matrix` is `True`, (batch_size,) otherwise.
        """
        return (
            -torch.einsum("id,jd->ij", x, y)
            if matrix
            else -torch.einsum("id,id->i", x, y)
        )
