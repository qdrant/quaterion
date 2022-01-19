from torch import Tensor, LongTensor
import torch.nn.functional as F
from quaterion.loss.group_loss import GroupLoss


class SoftmaxLoss(GroupLoss):
    """
    Regular cross-entropy loss.

    It is designed to work with the base `GroupLoss` class.
    """

    def __init__(self):
        super(GroupLoss, self).__init__()

    def forward(self, embeddings: Tensor, groups: LongTensor) -> Tensor:
        """
        :param embeddings: Shape: (batch_size, num_groups) - Here it represents logits from a suitable `EncoderHead`.
        :param groups: Shape: (batch_size,) - Group Groups, associated with embeddings.
        :return: 0-size tensor
        """
        return F.cross_entropy(embeddings, groups)
