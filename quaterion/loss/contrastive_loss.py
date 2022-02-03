from typing import Type

from torch import Tensor, relu, LongTensor

from quaterion.loss.metrics import SiameseDistanceMetric
from quaterion.loss.pairwise_loss import PairwiseLoss
from quaterion.utils import max_value_of_dtype


class ContrastiveLoss(PairwiseLoss):
    """
    Contrastive loss. Expects as input two texts and a label of either 0 or 1.
    If the label == 1, then the distance between the two embeddings is reduced. If the label == 0,
     then the distance between the embeddings is increased.

    Further information: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

    :param distance_metric_name: Name of the function, that returns a distance between two embeddings.
        The class SiameseDistanceMetric contains pre-defined metrics that can be used
    :param margin: Negative samples (label == 0) should have a distance of at least the margin value.
    :param size_average: Average by the size of the mini-batch.

    """

    @classmethod
    def metric_class(cls) -> Type:
        return SiameseDistanceMetric

    def __init__(
        self,
        distance_metric_name: str = "cosine_distance",
        margin: float = 1.0,
        size_average: bool = True,
    ):
        super().__init__(distance_metric_name=distance_metric_name)
        self.margin = margin
        self.size_average = size_average

    def get_config_dict(self):
        return {
            **super().get_config_dict(),
            "margin": self.margin,
            "size_average": self.size_average,
        }

    def forward(
        self,
        embeddings: Tensor,
        pairs: LongTensor,
        labels: Tensor,
        subgroups: Tensor,
        **kwargs
    ):
        rep_anchor = embeddings[pairs[:, 0]]
        rep_other = embeddings[pairs[:, 1]]
        distances = self.distance_metric(rep_anchor, rep_other)
        negative_distances_impact = 0.0

        if len(subgroups.unique()) > 1:
            embeddings_count = embeddings.shape[0]
            subgroup_matrix: Tensor = subgroups.repeat(embeddings_count, 1)
            comp_matrix: Tensor = subgroup_matrix != subgroup_matrix.T
            # a matrix to take into account only distances to negative
            # examples, i.e. from examples which don't belong to current
            # subgroup

            distance_matrix = self.distance_metric(embeddings, embeddings, matrix=True)
            distance_matrix[~comp_matrix] = max_value_of_dtype(distance_matrix.dtype)
            negative_distances, _ = distance_matrix.min(dim=1)  # find negative examples
            # which are the closest to positive ones
            neg_dist_to_anchors = negative_distances[pairs[:, 0]]
            neg_dist_to_other = negative_distances[pairs[:, 1]]
            negative_distances_impact = relu(self.margin - neg_dist_to_anchors).pow(
                2
            ) + relu(self.margin - neg_dist_to_other).pow(2)

        losses = (
            0.5
            * (
                labels.float() * distances.pow(2)
                + (1 - labels).float() * relu(self.margin - distances).pow(2)
            )
            + negative_distances_impact
        )
        return losses.mean() if self.size_average else losses.sum()
