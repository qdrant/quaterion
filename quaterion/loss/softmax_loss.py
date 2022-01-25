import torch
import torch.nn as nn
import torch.nn.functional as F
from quaterion.loss.group_loss import GroupLoss
from torch import LongTensor, Tensor


class SoftmaxLoss(GroupLoss):
    """
    Regular cross-entropy loss.

    It is designed to work with the base `GroupLoss` class.
    """

    def __init__(self, embedding_size: int, num_groups: int, temperature: float = 0.05):
        """An implementation of softmax with dot product.

        :param embedding_size: Output dimension of the encoder.
        :param num_groups: Number of groups in the dataset.
        :param temperature: Temperature value to divide logits, defaults to 0.05

        """
        super(GroupLoss, self).__init__()

        self.temperature = temperature
        self.kernel = nn.Parameter(torch.FloatTensor(embedding_size, num_groups))
        nn.init.normal_(self.kernel, std=0.01)

    def forward(self, embeddings: Tensor, groups: LongTensor) -> Tensor:
        """
        :param embeddings: Shape: (batch_size, vector_length) - Output embeddings from the encoder.
        :param groups: Shape: (batch_size,) - Group ids, associated with embeddings.
        :return: 0-size tensor
        """
        # Shape: (batch_size, num_groups)
        logits = torch.mm(embeddings, self.kernel) / self.temperature

        return F.cross_entropy(logits, groups)
