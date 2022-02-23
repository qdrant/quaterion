import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from quaterion.loss.group_loss import GroupLoss


def l2_norm(inputs: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Apply L2 normalization to tensor

    Args:
        inputs: Input tensor.
        dim: Dimension to operate on.

    Returns:
        torch.Tensor: L2-normalized tensor
    """
    outputs = inputs / torch.norm(inputs, 2, dim, True)

    return outputs


class ArcFaceLoss(GroupLoss):
    """Additive Angular Margin Loss as defined in https://arxiv.org/abs/1801.07698

    Args:
        embedding_size: Output dimension of the encoder.
        num_groups: Number of groups in the dataset.
        scale: Scaling value to make cross entropy work.
        margin: Margin value to push groups apart.
    """

    def __init__(
        self,
        embedding_size: int,
        num_groups: int,
        scale: float = 64.0,
        margin: float = 0.5,
    ):
        super(GroupLoss, self).__init__()

        self.kernel = nn.Parameter(torch.FloatTensor(embedding_size, num_groups))
        nn.init.normal_(self.kernel, std=0.01)
        self.scale = scale
        self.margin = margin

    def forward(self, embeddings, groups) -> Tensor:
        """Compute loss value

        Args:
            embeddings: shape: (batch_size, vector_length) - Output embeddings from the
                encoder.
            groups: shape: (batch_size,) - Group ids associated with embeddings.

        Returns:
            Tensor: loss value.
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
        m_hot.scatter_(1, groups[index, None], self.margin)
        cos_theta.acos_()
        cos_theta[index] += m_hot
        cos_theta.cos_().mul_(self.scale)

        # calculate scalar loss
        loss = F.cross_entropy(cos_theta, groups)

        return loss
