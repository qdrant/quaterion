from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import LongTensor, Tensor

from quaterion.loss.group_loss import GroupLoss
from quaterion.utils import l2_norm


class CosFaceLoss(GroupLoss):

    """Large Margin Cosine Loss as defined in https://arxiv.org/pdf/1801.09414.pdf

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
        margin: Optional[float] = 0.35,
        scale: Optional[float] = 64.0,
    ):
        super(GroupLoss, self).__init__()

        self.kernel = nn.Parameter(torch.FloatTensor(embedding_size, num_groups))
        nn.init.normal_(self.kernel, std=0.01)
        self.scale = scale
        self.margin = margin

    def forward(self, embeddings: Tensor, groups: LongTensor) -> Tensor:
        """Compute loss value
        Args:
            embeddings: shape: (batch_size, vector_length) - Output embeddings from the
                encoder.
            groups: shape: (batch_size,) - Group ids associated with embeddings.
        Returns:
            Tensor: loss value.
        """
        assert (
            groups.ge(0).all() and groups.lt(self.kernel.size(1)).all()
        ), f"Invalid group ids: all the values must be between 0 (inclusive) and num_groups (exclusive), but given:  {groups}"

        embeddings = l2_norm(embeddings, 1)
        kernel_norm = l2_norm(self.kernel, 0)

        # Shape: (batch_size, num_groups)
        cos_theta = torch.mm(embeddings, kernel_norm)
        # insure numerical stability
        cos_theta = cos_theta.clamp(-1, 1)

        # Shape: (batch_size,)
        index = torch.where(groups != -1)[0]

        # Shape: (batch_size, num_groups)
        margins = torch.zeros(
            index.size()[0], cos_theta.size()[1], device=cos_theta.device
        )
        margins.scatter_(1, groups[index, None], self.margin)

        cos_theta[index] -= margins

        cos_theta.cos_().mul_(self.scale)

        # calculate scalar loss
        loss = F.cross_entropy(cos_theta, groups)

        return loss
