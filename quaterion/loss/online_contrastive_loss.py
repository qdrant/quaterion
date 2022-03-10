from typing import Optional

import torch
import torch.nn.functional as F
from quaterion.loss.group_loss import GroupLoss
from quaterion.loss.metrics import SiameseDistanceMetric
from quaterion.utils import (
    max_value_of_dtype,
    get_anchor_positive_mask,
    get_anchor_negative_mask,
)


class OnlineContrastiveLoss(GroupLoss):
    """Implements Contrastive Loss as defined in http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

    Unlike `quaterion.loss.ContrastiveLoss`, this one supports batch-all and batch-hard
    strategies for online pair mining.

    Args:
        margin: Margin value to push negative examples
            apart. Optional, defaults to `0.5`.
        distance_metric_name: Name of the distance function. Optional, defaults to `euclidean`.
        squared (bool, optional): Squared Euclidean distance or not. Defaults to `True`.
        mining (str, optional): Pair mining strategy. One of
            `"all"`, `"hard"`. Defaults to `"hard"`.
    """

    def __init__(
        self,
        margin: Optional[float] = 0.5,
        distance_metric_name: str = "euclidean",
        squared: Optional[bool] = True,
        mining: Optional[str] = "hard",
    ):
        distance_metrics = ["euclidean", "cosine_distance"]
        if distance_metric_name not in distance_metrics:
            raise ValueError(
                f"Not supported distance metrc for this loss: {distance_metric_name}. "
                f"Must be one of {', '.join(distance_metrics)}"
            )

        mining_types = ["all", "hard"]
        if mining not in mining_types:
            raise ValueError(
                f"Unrecognized mining strategy: {mining}. Must be one of {', '.join(mining_types)}"
            )
        super(OnlineContrastiveLoss, self).__init__(
            distance_metric_name=distance_metric_name
        )

        self._margin = margin
        self._squared = squared
        self._mining = mining

    def get_config_dict(self):
        config = super().get_config_dict()
        config.update(
            {
                "margin": self._margin,
                "squared": self._squared,
                "mining": self._mining,
            }
        )
        return config

    def forward(
        self, embeddings: torch.Tensor, groups: torch.LongTensor
    ) -> torch.Tensor:
        """Calculates Triplet Loss with specified embeddings and labels.

        Args:
            embeddings (torch.Tensor): Batch of embeddings. Shape: (batch_size, embedding_dim)
            groups (torch.LongTensor): Batch of labels associated with `embeddings`. Shape: (batch_size,)

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Shape: (batch_size, batch_size)
        dists = (
            SiameseDistanceMetric.euclidean(
                x=embeddings, matrix=True, squared=self._squared
            )
            if self.distance_metric_name == "euclidean"
            else SiameseDistanceMetric.cosine_distance(x=embeddings, matrix=True)
        )

        if self._mining == "all":

            # get a mask for valid anchor-positive pairs and calculate the number of them
            anchor_positive_mask = get_anchor_positive_mask(groups).float()
            num_positive_pairs = anchor_positive_mask.sum()

            anchor_positive_dists = (
                anchor_positive_mask * dists
            )  # invalid pairs set to 0
            positive_loss = anchor_positive_dists.pow(2).sum() / torch.max(
                num_positive_pairs, torch.tensor(1e-16)
            )

            # get a mask for valid anchor-negative pairs, and set invalid ones to a maximum value
            # to not count them later on
            anchor_negative_mask = get_anchor_negative_mask(groups)
            anchor_negative_dists = dists
            anchor_negative_dists[~anchor_negative_mask] = max_value_of_dtype(
                anchor_negative_dists.dtype
            )
            num_negative_pairs = anchor_negative_mask.float().sum()

            negative_loss = F.relu(self._margin - anchor_negative_dists).pow(
                2
            ).sum() / torch.max(num_negative_pairs, torch.tensor(1e-16))

            total_loss = 0.5 * (positive_loss + negative_loss)
        else:  # batch-hard pair mining
            # TODO: Implement batch-hard strategy for online pair mining
            total_loss = 0

        return total_loss
