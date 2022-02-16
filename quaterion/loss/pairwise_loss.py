from torch import Tensor

from quaterion.loss.similarity_loss import SimilarityLoss


class PairwiseLoss(SimilarityLoss):
    """Base class for pairwise losses.

    Args:
        distance_metric_name: Name of the function, that returns a distance between two
            embeddings. The class SiameseDistanceMetric contains pre-defined metrics
            that can be used.
    """

    def __init__(self, distance_metric_name: str = "cosine_distance"):
        super(PairwiseLoss, self).__init__(distance_metric_name=distance_metric_name)

    def forward(
        self, embeddings: Tensor, pairs: Tensor, labels: Tensor, subgroups: Tensor
    ) -> Tensor:
        """Compute loss value.

        Args:
            embeddings: shape: (batch_size, vector_length)
            pairs: shape: (2 * pairs_count,) - contains a list of known similarity pairs
                in batch
            labels: shape: (pairs_count,) - similarity of the pair
            subgroups: shape: (2 * pairs_count,) - subgroup ids of objects

        Returns:
            Tensor: zero-size tensor, loss value
        """
        raise NotImplementedError()
