from torch import Tensor, LongTensor
import torch.nn.functional as F
from quaterion.loss.group_loss import GroupLoss


class SoftmaxLoss(GroupLoss):
    def __init__(self):
        super(GroupLoss, self).__init__()

    def forward(self, embeddings: Tensor, groups: LongTensor) -> Tensor:
        """
        :param embeddings: shape: [batch_size, num_groups]
        :param groups: shape: [batch_size] - Group Groups, associated with embeddings
        :return: 0-size tensor
        """
        return F.cross_entropy(embeddings, groups)
