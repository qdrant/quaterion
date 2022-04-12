from typing import Optional

from torch import Tensor


class BaseDistance:
    """Provides a base class that any distance metric should implement."""

    @staticmethod
    def distance(x: Tensor, y: Tensor) -> Tensor:
        """Calculate distances, i.e., the lower the value, the more similar the samples.

        Args:
            x: shape: (batch_size, embedding_dim)
            y: shape: (batch_size, embedding_dim)

        Returns:
            Distances - shape: (batch_size,)
        """
        raise NotImplementedError

    @staticmethod
    def similarity(x: Tensor, y: Tensor) -> Tensor:
        """Calculate similarities, i.e., the higher the value, the more similar the samples.

        Args:
            x: shape: (batch_size, embedding_dim)
            y: shape: (batch_size, embedding_dim)

        Returns:
            Similarities - shape: (batch_size,)
        """
        raise NotImplementedError

    @staticmethod
    def distance_matrix(x: Tensor, y: Optional[Tensor] = None) -> Tensor:
        """Calculate a distance matrix, i.e., distances between all possible pairs in `x` and `y`.

        Args:
            x: shape: (batch_size, embedding_dim)
            y: shape: (batch_size, embedding_dim). If `y is None`, it assigns `x` to `y`.

        Returns:
            Distance matrix - shape: (batch_size, batch_size)
        """
        raise NotImplementedError

    @staticmethod
    def similarity_matrix(x: Tensor, y: Optional[Tensor] = None) -> Tensor:
        """Calculate a similarity matrix, i.e., similarities between all possible pairs in `x` and `y`.

        Args:
            x: shape: (batch_size, embedding_dim)
            y: shape: (batch_size, embedding_dim). If `y is None`, it assigns `x` to `y`.

        Returns:
            Similarity matrix - shape: (batch_size, batch_size)
        """
        raise NotImplementedError
