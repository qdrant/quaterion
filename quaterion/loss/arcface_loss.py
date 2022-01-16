import torch
import torch.nn as nn
import torch.nn.functional as F
from quaterion.loss.group_loss import GroupLoss


def l2_norm(inputs, dim=0):
    outputs = inputs / torch.norm(inputs, 2, dim, True)
    return outputs


class ArcfaceLoss(GroupLoss):
    def __init__(self, embedding_size: int, num_classes: int, s: float = 64., m: float = 0.5):
        super(GroupLoss, self).__init__()

        self.kernel = nn.Parameter(
            torch.FloatTensor(embedding_size, num_classes))
        nn.init.normal_(self.kernel, std=0.01)
        self.num_classes = num_classes
        self.s = s
        self.m = m

    def forward(self, embeddings, groups):
        embeddings = l2_norm(embeddings, 1)
        kernel_norm = l2_norm(self.kernel, 0)
        cos_theta = torch.mm(embeddings, kernel_norm).clamp(-1, 1)
        index = torch.where(groups != -1)[0]
        m_hot = torch.zeros(index.size()[0], cos_theta.size()[1], device=cos_theta.device)
        m_hot.scatter_(1, groups[index, None], self.m)
        cos_theta.acos_()
        cos_theta[index] += m_hot
        cos_theta.cos_().mul_(self.s)
        
        loss = F.cross_entropy(cos_theta, groups)

        return loss
