from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from quaterion.distances import Distance
from quaterion.loss.group_loss import GroupLoss
from quaterion.utils import get_anchor_negative_mask, get_anchor_positive_mask


class FastAPLoss(GroupLoss):
    """FastAP Loss

    Adaptation from https://github.com/kunhe/FastAP-metric-learning.

    Further information:
        https://cs-people.bu.edu/fcakir/papers/fastap_cvpr2019.pdf.
        "Deep Metric Learning to Rank"
        Fatih Cakir(*), Kun He(*), Xide Xia, Brian Kulis, and Stan Sclaroff
        IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019

    Args:
        num_bins:The number of soft histogram bins for calculating average precision. The paper suggests using 10.
    """

    def __init__(self, num_bins: Optional[int] = 10):
        # Eucledian distance is the only compatible distance metric for FastAP Loss
        super(GroupLoss, self).__init__(distance_metric_name=Distance.EUCLIDEAN)
        self.num_bins = num_bins

    def get_config_dict(self) -> Dict[str, Any]:
        """Config used in saving and loading purposes.

        Config object has to be JSON-serializable.

        Returns:
            Dict[str, Any]: JSON-serializable dict of params
        """
        config = self.get_config_dict()
        config.update(
            {
                "num_bins": self.num_bins,
                "distance_metric_name": self.distance_metric_name,
            }
        )

        return config

    def forward(
        self,
        embeddings: Tensor,
        groups: Tensor,
    ) -> Tensor:
        """Compute loss value.

        Args:
            embeddings: shape: (batch_size, vector_length) - Batch of embeddings.
            groups: shape: (batch_size,) - Batch of labels associated with `embeddings`.
        Returns:
            Tensor: Scalar loss value.
        """

        _warn = "Batch size of embeddings and groups don't match."

        batch_size = groups.size()[0]  # batch size
        assert embeddings.size()[0] == batch_size, _warn

        # 1. get positive and negative masks
        pos_mask = get_anchor_positive_mask(groups)  # (batch_size, batch_size)
        neg_mask = get_anchor_negative_mask(groups)  # (batch_size, batch_size)
        n_pos = torch.sum(pos_mask, dim=1)  # Sum over all columns (for each row)

        # 2. compute distances from embeddings squared Euclidean distance matrix
        embeddings = F.normalize(embeddings, p=2, dim=1)  # normalize embeddings
        dist_matrix = (
            self.distance_metric.distance_matrix(embeddings) ** 2
        )  # (batch_size, batch_size)

        # 3. estimate discrete histograms
        histogram_delta = torch.tensor(4.0 / self.num_bins)
        mid_points = torch.linspace(0.0, 4.0, steps=self.num_bins + 1).view(-1, 1, 1)

        pulse = F.relu(
            input=1 - torch.abs(dist_matrix - mid_points) / histogram_delta
        )  # max(0, input)

        pos_hist = torch.t(torch.sum(pulse * pos_mask, dim=2))  # positive histograms
        neg_hist = torch.t(torch.sum(pulse * neg_mask, dim=2))  # negative histograms

        total_pos_hist = torch.cumsum(pos_hist, dim=1)
        total_hist = torch.cumsum(pos_hist + neg_hist, dim=1)

        # 4. compute FastAP
        FastAP = pos_hist * total_pos_hist / total_hist
        FastAP[torch.isnan(FastAP) | torch.isinf(FastAP)] = 0
        FastAP = torch.sum(FastAP, 1) / n_pos
        FastAP = FastAP[~torch.isnan(FastAP)]
        loss = 1 - torch.mean(FastAP)

        return loss
