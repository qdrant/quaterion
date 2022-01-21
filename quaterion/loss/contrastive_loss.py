from typing import Type

from torch import Tensor, relu, LongTensor

from quaterion.loss.metrics import SiameseDistanceMetric
from quaterion.loss.pairwise_loss import PairwiseLoss


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
        super(ContrastiveLoss, self).__init__(distance_metric_name=distance_metric_name)
        self.margin = margin
        self.size_average = size_average

    def get_config_dict(self):
        return {
            **super().get_config_dict(),
            "margin": self.margin,
            "size_average": self.size_average,
        }

    def forward(self, embeddings: Tensor, pairs: LongTensor, labels: Tensor, **kwargs):
        rep_anchor = embeddings[pairs[:, 0]]
        rep_other = embeddings[pairs[:, 1]]
        distances = self.distance_metric(rep_anchor, rep_other)
        losses = 0.5 * (
            labels.float() * distances.pow(2)
            + (1 - labels).float() * relu(self.margin - distances).pow(2)
        )
        return losses.mean() if self.size_average else losses.sum()
