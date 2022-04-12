from torch import Tensor, LongTensor

from quaterion.distances import Distance
from quaterion.loss.similarity_loss import SimilarityLoss


class GroupLoss(SimilarityLoss):
    """Base class for Group losses.

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
            groups: shape: (batch_size,) - Groups, associated with embeddings

        Returns:
            Tensor: zero-size tensor, loss value
        """
        raise NotImplementedError()
