from torch import Tensor, LongTensor

from quaterion.loss.similarity_loss import SimilarityLoss


class GroupLoss(SimilarityLoss):
    def __init__(self, distance_metric_name: str = "cosine_distance"):
        super(GroupLoss, self).__init__(distance_metric_name=distance_metric_name)

    def forward(self, embeddings: Tensor, groups: LongTensor) -> Tensor:
        """
        :param embeddings: shape: [batch_size x vector_length]
        :param groups: shape: [batch_size] - Groups, associated with embeddings
        :return: 0-size tensor
        """
        raise NotImplementedError()
