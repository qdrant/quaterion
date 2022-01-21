from torch import Tensor

from quaterion.loss.similarity_loss import SimilarityLoss


class PairwiseLoss(SimilarityLoss):
    def __init__(self, distance_metric_name: str = "cosine_distance"):
        super(PairwiseLoss, self).__init__(distance_metric_name=distance_metric_name)

    def forward(
        self, embeddings: Tensor, pairs: Tensor, labels: Tensor, subgroups: Tensor
    ) -> Tensor:
        """
        :param embeddings: shape: [batch_size x vector_length]
        :param pairs: shape: [pairs_count x 2] - contains a list of known similarity pairs in batch
        :param labels: shape: [pairs_count] - similarity of the pair
        :param subgroups: shape: [pairs_count x 2] - subgroup ids of objects
        :return:
        """
        raise NotImplementedError()
