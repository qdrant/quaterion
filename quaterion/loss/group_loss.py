from torch import Tensor, LongTensor

from quaterion.loss.similarity_loss import SimilarityLoss


class GroupLoss(SimilarityLoss):
    """Base class for Group losses.

    Args:
        distance_metric_name: Name of the function, that returns a distance between two embeddings.
            :class:`~quaterion.loss.metrics.SiameseDistanceMetric` contains pre-defined metrics
            that can be used.

    """

    def __init__(self, distance_metric_name: str = "cosine_distance"):
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
