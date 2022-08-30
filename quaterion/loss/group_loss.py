from typing import Optional

from torch import LongTensor, Tensor

from quaterion.distances import Distance
from quaterion.loss.similarity_loss import SimilarityLoss


class GroupLoss(SimilarityLoss):
    """Base class for group losses.

    Args:
        distance_metric_name: Name of the distance function, e.g.,
            :class:`~quaterion.distances.Distance`.
    """

    def __init__(self, distance_metric_name: Distance = Distance.COSINE):
        super(GroupLoss, self).__init__(distance_metric_name=distance_metric_name)

    def forward(self, embeddings: Tensor, groups: LongTensor) -> Tensor:
        """

        Args:
            embeddings: shape: (batch_size, vector_length)
            groups: shape: (batch_size,) - Groups, associated with `embeddings`

        Returns:
            Tensor: zero-size tensor, loss value
        """
        raise NotImplementedError()

    def xbm_loss(
        self,
        embeddings: Tensor,
        groups: LongTensor,
        memory_embeddings: Tensor,
        memory_groups: LongTensor,
    ) -> Tensor:
        """Implement XBM loss computation for this loss.

        Args:
            embeddings: shape: (batch_size, vector_length) - Output embeddings from the
                encoder.
            groups: shape: (batch_size,) - Group ids associated with embeddings.
            memory_embeddings: shape: (memory_buffer_size, vector_length) - Embeddings stored
                in a ring buffer
            memory_groups: shape: (memory_buffer_size,) - Groups ids associated with `memory_embeddings`

        Returns:
            Tensor: zero-size tensor, XBM loss value.
        """
        raise NotImplementedError(
            f"XBM is not implemented for {self.__class__.__name__}"
        )
