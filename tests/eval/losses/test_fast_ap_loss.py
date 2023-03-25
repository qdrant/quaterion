from typing import Any, Optional

import torch
import torch.nn.functional as F

from quaterion.loss import FastAPLoss

####################################
#    Official Implementation       #
####################################
# From https://github.com/kunhe/FastAP-metric-learning/blob/master/pytorch/FastAP_loss.py
# This code is copied from the official implementation to compare our results. It's copied under the MIT license.


def soft_binning(
    D: torch.Tensor, mid: torch.Tensor, Delta: torch.Tensor
) -> torch.Tensor:
    y = 1 - torch.abs(D - mid) / Delta
    return torch.max(torch.tensor([0], dtype=D.dtype).to(D.device), y)


class OfficialFastAP(torch.autograd.Function):
    """
    FastAP - autograd function definition

    This class implements the FastAP loss from the following paper:
    "Deep Metric Learning to Rank",
    F. Cakir, K. He, X. Xia, B. Kulis, S. Sclaroff. CVPR 2019
    """

    @staticmethod
    def forward(
        ctx: Any, input: torch.Tensor, target: torch.Tensor, num_bins: int
    ) -> torch.Tensor:
        """
        Args:
            input:     torch.Tensor(N x embed_dim), embedding matrix
            target:    torch.Tensor(N x 1), class labels
            num_bins:  int, number of bins in distance histogram
        """
        N = target.size()[0]
        assert input.size()[0] == N, "Batch size doesn't match!"

        # 1. get affinity matrix
        Y = target.unsqueeze(1)
        Aff = 2 * (Y == Y.t()).type(input.dtype) - 1
        Aff.masked_fill_(
            torch.eye(N, N).bool().to(input.device), 0
        )  # set diagonal to 0

        I_pos = (Aff > 0).type(input.dtype).to(input.device)
        I_neg = (Aff < 0).type(input.dtype).to(input.device)
        N_pos = torch.sum(I_pos, 1)

        # 2. compute distances from embeddings
        # squared Euclidean distance with range [0,4]
        dist2 = 2 - 2 * torch.mm(input, input.t())
        # 3. estimate discrete histograms
        Delta = torch.tensor(4.0 / num_bins).to(input.device)
        Z = torch.linspace(0.0, 4.0, steps=num_bins + 1).to(input.device)
        L = Z.size()[0]
        h_pos = torch.zeros((N, L), dtype=input.dtype).to(input.device)
        h_neg = torch.zeros((N, L), dtype=input.dtype).to(input.device)
        for idx in range(L):
            pulse = soft_binning(dist2, Z[idx], Delta)
            h_pos[:, idx] = torch.sum(pulse * I_pos, 1)
            h_neg[:, idx] = torch.sum(pulse * I_neg, 1)

        H_pos = torch.cumsum(h_pos, 1)
        h = h_pos + h_neg
        H = torch.cumsum(h, 1)

        # 4. compate FastAP
        FastAP = h_pos * H_pos / H
        FastAP[torch.isnan(FastAP) | torch.isinf(FastAP)] = 0
        FastAP = torch.sum(FastAP, 1) / N_pos
        FastAP = FastAP[~torch.isnan(FastAP)]
        loss = 1 - torch.mean(FastAP)

        return loss


class OfficialFastAPLoss(torch.nn.Module):
    """
    FastAP - loss layer definition

    This class implements the FastAP loss from the following paper:
    "Deep Metric Learning to Rank",
    F. Cakir, K. He, X. Xia, B. Kulis, S. Sclaroff. CVPR 2019
    """

    def __init__(self, num_bins: Optional[int] = 10):
        super(OfficialFastAPLoss, self).__init__()
        self.num_bins = num_bins

    def forward(self, batch: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return OfficialFastAP.apply(batch, labels, self.num_bins)


class TestFastAPLoss:
    embeddings = torch.Tensor(
        [
            [0.0, -1.0, 0.5],
            [0.1, 2.0, 0.5],
            [0.0, 2.3, 0.2],
            [1.0, 0.0, 0.9],
            [1.2, -1.2, 0.01],
            [-0.7, 0.0, 1.5],
        ]
    )

    groups = torch.Tensor([1, 2, 3, 3, 2, 1])

    def test_batch_all(self):
        num_bins = 5
        loss = FastAPLoss(num_bins)

        actual_loss = loss.forward(embeddings=self.embeddings, groups=self.groups)

        assert actual_loss.shape == torch.Size([])

        expected_loss = OfficialFastAPLoss(num_bins)(
            F.normalize(self.embeddings), labels=self.groups
        )

        rtol = 1e-2 if torch.dtype == torch.float16 else 1e-5
        assert torch.isclose(expected_loss, actual_loss, rtol=rtol)
