from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import LongTensor, Tensor

from quaterion.loss.group_loss import GroupLoss


class SoftmaxLoss(GroupLoss):
    """Regular cross-entropy loss.

    An implementation of softmax with dot product.
    It is designed to work with the base :class:`~quaterion.loss.group_loss.GroupLoss`.

    Args:
        embedding_size: Output dimension of the encoder.
        num_groups: Number of groups in the dataset.
        temperature: Temperature value to divide logits, defaults to 0.05

    """

    def __init__(self, embedding_size: int, num_groups: int, temperature: float = 0.05):
        super(GroupLoss, self).__init__()
        self.temperature = temperature
        self.kernel = nn.Parameter(torch.FloatTensor(embedding_size, num_groups))
        nn.init.normal_(self.kernel, std=0.01)

    def forward(
        self,
        embeddings: Tensor,
        groups: LongTensor,
        memory_embeddings: Optional[Tensor] = None,
        memory_groups: Optional[LongTensor] = None,
    ) -> Tensor:
        """Compute loss value.

        Args:
            embeddings: shape: (batch_size, vector_length) - Output embeddings from the
                encoder
            groups: shape: (batch_size,) - Group ids, associated with embeddings
            memory_embeddings: shape: (memory_buffer_size, vector_length) - Embeddings stored
                in a ring buffer. Used only for XBM
            memory_groups: shape: (memory_buffer_size,) - Groups ids associated with `memory_embeddings`.
                Used only for XBM

        Returns:
            Tensor: zero-size tensor, loss value
        """
        if memory_embeddings is not None or memory_groups is not None:
            return self._compute_xbm_loss(
                embeddings, groups, memory_embeddings, memory_groups
            )

        # shape: (batch_size, num_groups)
        logits = torch.mm(embeddings, self.kernel) / self.temperature

        return F.cross_entropy(logits, groups)
