from typing import Optional

import torch
import torch.nn.functional as F
from torch import LongTensor, Tensor

from quaterion.distances import Distance
from quaterion.loss.group_loss import GroupLoss
from quaterion.utils import get_anchor_negative_mask, get_anchor_positive_mask


class CircleLoss(GroupLoss):
    """Implements Circle Loss as defined in https://arxiv.org/abs/2002.10857.

    Args:
        margin: Margin value to push negative examples.
        scale_factor: scale factor γ determines the largest scale of each similarity score.

    Note:
        Refer to sections 4.1 and 4.5 in the paper for default values and evaluation of margin and scaling_factor hyperparameters.
    """

    def __init__(
        self,
        margin: Optional[float] = 0.25,
        scale_factor: Optional[float] = 256,
        distance_metric_name: Optional[Distance] = Distance.COSINE,
    ):
        super(GroupLoss, self).__init__(distance_metric_name=distance_metric_name)
        self.margin = margin
        self.scale_factor = scale_factor
        self.op = 1 + self.margin
        self.on = -self.margin
        self.delta_positive = 1 - self.margin
        self.delta_negative = self.margin

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
        similarity_matrix = self.distance_metric.similarity_matrix(embeddings)

        # Shape: (batch_size * batch_size,)
        similarity_matrix = torch.reshape(similarity_matrix, [-1])

        # Shape: (batch_size, batch_size)
        pos_mask = get_anchor_positive_mask(groups, groups).triu(diagonal=1)
        # Shape: (batch_size, batch_size)
        neg_mask = get_anchor_negative_mask(groups, groups).triu(diagonal=1)

        # Shape: (batch_size * batch_size,)
        pos_mask = torch.reshape(pos_mask, [-1])
        # Shape: (batch_size * batch_size,)
        neg_mask = torch.reshape(neg_mask, [-1])

        sp = similarity_matrix[pos_mask]
        sn = similarity_matrix[neg_mask]

        # get alpha-positive and alpha-negative weights as described in https://arxiv.org/abs/2002.10857.
        ap = torch.clamp_min(self.op + sp.detach(), min=0)
        an = torch.clamp_min(self.on + sn.detach(), min=0)

        exp_p = -ap * self.scale_factor * (sp - self.delta_positive)
        exp_n = an * self.scale_factor * (sn - self.delta_negative)

        circle_loss = F.softplus(
            torch.logsumexp(exp_n, dim=0) + torch.logsumexp(exp_p, dim=0)
        )

        return circle_loss
