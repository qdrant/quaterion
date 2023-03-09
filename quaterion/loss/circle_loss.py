from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import LongTensor, Tensor

from quaterion.loss.group_loss import GroupLoss


class CircleLoss(GroupLoss):
    """Implements Circle Loss as defined in https://arxiv.org/abs/2002.10857.

    Args:
        margin: Margin value to push negative examples.
        scale_factor: scale factor γ determines the largest scale of each similarity score.
    """

    def __init__(
        self,
        margin: Optional[float],
        scale_factor: Optional[float],
        distance_metric_name: Optional[Distance] = Distance.COSINE,
    ):
        super(GroupLoss, self).__init__()
        self.margin = margin
        self.scale_factor = scale_factor
        self.op = 1 + self._margin
        self.on = -self._margin
        self.delta_positive = 1 - self._margin
        self.delta_negative = self._margin

    def forward(
        self,
        embeddings: Tensor,
        groups: LongTensor,
    ) -> Tensor:
        """Compute loss value.

        Args:
            embeddings: shape: (batch_size, vector_length) - Batch of embeddings.
            groups: shape: (batch_size,) - Batch of labels associated with `embeddings`

        Returns:
            Tensor: Scalar loss value.
        """
        # Shape: (batch_size, batch_size)
        dists = self.distance_metric.distance_matrix(embeddings)
        # Calculate loss for all possible triplets first, then filter by group mask
        # Shape: (batch_size, batch_size, 1)
        sp = dists.unsqueeze(2)
        # Shape: (batch_size, 1, batch_size)
        sn = dists.unsqueeze(1)
        # get alpha-positive and alpha-negative weights as described in https://arxiv.org/abs/2002.10857.
        ap = torch.clamp_min(self.op + sp.detach(), min=0)
        an = torch.clamp_min(self.on + sn.detach(), min=0)

        exp_p = -ap * self.scale_factor * (sp - self.delta_positive)
        exp_n = an * self.scale_factor * (sn - self.delta_negative)

        circle_loss = F.softplus(
            torch.logsumexp(exp_n, dim=0) + torch.logsumexp(exp_p, dim=0)
        )

        return circle_loss
