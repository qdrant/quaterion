from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import LongTensor, Tensor

from quaterion.loss.group_loss import GroupLoss
from quaterion.utils import l2_norm


class CenterLoss(GroupLoss):
    """
    Center Loss as defined in the paper "A Discriminative Feature Learning Approach
    for Deep Face Recognition" (http://ydwen.github.io/papers/WenECCV16.pdf)
    It aims to minimize the intra-class variations while keeping the features of
    different classes separable.

    Args:
        embedding_size: Output dimension of the encoder.
        num_groups: Number of groups (classes) in the dataset.
        lambda_c: A regularization parameter that controls the contribution of the center loss.
    """

    def __init__(
        self, embedding_size: int, num_groups: int, lambda_c: Optional[float] = 0.5
    ):
        super(GroupLoss, self).__init__()
        self.num_groups = num_groups
        self.centers = nn.Parameter(torch.randn(num_groups, embedding_size))
        self.lambda_c = lambda_c

        nn.init.xavier_uniform_(self.centers)

    def forward(self, embeddings: Tensor, groups: LongTensor) -> Tensor:
        """
        Compute the Center Loss value.

        Args:
            embeddings: shape (batch_size, vector_length) - Output embeddings from the encoder.
            groups: shape (batch_size,) - Group (class) ids associated with embeddings.

        Returns:
            Tensor: loss value.
        """
        embeddings = l2_norm(embeddings, 1)

        # Gather the center for each embedding's corresponding group
        centers_batch = self.centers.index_select(0, groups)

        # Calculate the distance between embeddings and their respective class centers
        loss = F.mse_loss(embeddings, centers_batch)

        # Scale the loss by the regularization parameter
        loss *= self.lambda_c

        return loss
