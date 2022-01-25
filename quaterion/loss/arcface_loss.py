import torch
import torch.nn as nn
import torch.nn.functional as F
from quaterion.loss.group_loss import GroupLoss


def l2_norm(inputs: torch.Tensor, dim: int = 0):
    """
    :param inputs: Input tensor.
    :param dim: Dimension to operate on.
    :return: L2-normalized tensor.
    """
    outputs = inputs / torch.norm(inputs, 2, dim, True)

    return outputs


class ArcfaceLoss(GroupLoss):
    """
    Additive Angular Margin Loss as defined in https://arxiv.org/abs/1801.07698
    """

    def __init__(
        self, embedding_size: int, num_groups: int, s: float = 64.0, m: float = 0.5
    ):
        """
        :param embedding_size: Output dimension of the encoder.
        :param num_groups: Number of groups in the dataset.
        :param s: Scaling value to make cross entropy work.
        :param m: Margin value to push groups apart.

        """
        super(GroupLoss, self).__init__()

        self.kernel = nn.Parameter(torch.FloatTensor(embedding_size, num_groups))
        nn.init.normal_(self.kernel, std=0.01)
        self.s = s
        self.m = m

    def forward(self, embeddings, groups):
        """
        :param embeddings: Shape: (batch_size, vector_length) - Output embeddings from the encoder.
        :param groups: Shape: (batch_size,) - Group ids associated with embeddings.
        :return: Zero-size tensor.
        """
        embeddings = l2_norm(embeddings, 1)
        kernel_norm = l2_norm(self.kernel, 0)

        # Shape: (batch_size, num_groups)
        cos_theta = torch.mm(embeddings, kernel_norm)
        # insure numerical stability
        cos_theta = cos_theta.clamp(-1, 1)

        # Shape: (batch_size,)
        index = torch.where(groups != -1)[0]

        # Shape: (batch_size, num_groups)
        m_hot = torch.zeros(
            index.size()[0], cos_theta.size()[1], device=cos_theta.device
        )
        m_hot.scatter_(1, groups[index, None], self.m)
        cos_theta.acos_()
        cos_theta[index] += m_hot
        cos_theta.cos_().mul_(self.s)

        # calculate scalar loss
        loss = F.cross_entropy(cos_theta, groups)

        return loss
