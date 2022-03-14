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

    def forward(self, embeddings: Tensor, groups: LongTensor) -> Tensor:
        """Compute loss value.

        Args:
            embeddings: shape: (batch_size, vector_length) - Output embeddings from the
                encoder.
            groups: shape: (batch_size,) - Group ids, associated with embeddings.

        Returns:
            Tensor: zero-size tensor, loss value
        """
        # shape: (batch_size, num_groups)
        logits = torch.mm(embeddings, self.kernel) / self.temperature

        return F.cross_entropy(logits, groups)
