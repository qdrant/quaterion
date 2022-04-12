from typing import Optional

import torch
import torch.nn.functional as F
from quaterion.distances import Distance
from quaterion.loss.group_loss import GroupLoss
from quaterion.utils import (
    max_value_of_dtype,
    get_anchor_positive_mask,
    get_anchor_negative_mask,
)


class OnlineContrastiveLoss(GroupLoss):
    """Implements Contrastive Loss as defined in
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

    Unlike :class:`~quaterion.loss.contrastive_loss.ContrastiveLoss`, this one supports online pair
    mining, i.e., it makes positive and negative pairs on-the-fly, so you don't need to form such
    pairs yourself. Instead, it first calculates all possible pairs in a batch, and then filters
    valid positive pairs and valid negative pairs separately. Batch-all and batch-hard strategies
    for online pair mining are supported.

    Args:
        margin: Margin value to push negative examples
            apart. Optional, defaults to `0.5`.
        distance_metric_name: Name of the distance function, e.g.,
            :class:`~quaterion.distances.Distance`. Optional, defaults to
            :attr:`~quaterion.distances.Distance.COSINE`.
        mining (str, optional): Pair mining strategy. One of `"all"`, `"hard"`.
            Defaults to `"hard"`.
    """

    def __init__(
        self,
        margin: Optional[float] = 0.5,
        distance_metric_name: Distance = Distance.COSINE,
        mining: Optional[str] = "hard",
    ):
        mining_types = ["all", "hard"]
        if mining not in mining_types:
            raise ValueError(
                f"Unrecognized mining strategy: {mining}. Must be one of {', '.join(mining_types)}"
            )
        super(OnlineContrastiveLoss, self).__init__(
            distance_metric_name=distance_metric_name
        )

        self._margin = margin
        self._mining = mining

    def get_config_dict(self):
        config = super().get_config_dict()
        config.update(
            {
                "margin": self._margin,
                "mining": self._mining,
            }
        )

        return config

    def forward(
        self, embeddings: torch.Tensor, groups: torch.LongTensor
    ) -> torch.Tensor:
        """Calculates Contrastive Loss by making pairs on-the-fly.

        Args:
            embeddings (torch.Tensor): Batch of embeddings. Shape: (batch_size, embedding_dim)
            groups (torch.LongTensor): Batch of labels associated with `embeddings`.
                Shape: (batch_size,)

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Shape: (batch_size, batch_size)
        dists = self.distance_metric.distance_matrix(embeddings)

        # get a mask for valid anchor-positive pairs and apply it to the distance matrix
        # to set invalid ones to 0
        anchor_positive_mask = get_anchor_positive_mask(groups).float()

        anchor_positive_dists = anchor_positive_mask * dists  # invalid pairs set to 0

        # get a mask for valid anchor-negative pairs, and apply it to distance matrix
        # # to set invalid ones to a maximum value of dtype
        anchor_negative_mask = get_anchor_negative_mask(groups)
        anchor_negative_dists = dists
        anchor_negative_dists[~anchor_negative_mask] = max_value_of_dtype(
            anchor_negative_dists.dtype
        )

        if self._mining == "all":

            num_positive_pairs = anchor_positive_mask.sum()
            positive_loss = anchor_positive_dists.sum() / torch.max(
                num_positive_pairs, torch.tensor(1e-16)
            )

            num_negative_pairs = anchor_negative_mask.float().sum()

            negative_loss = F.relu(
                self._margin - anchor_negative_dists
            ).sum() / torch.max(num_negative_pairs, torch.tensor(1e-16))

        else:  # batch-hard pair mining

            # get the hardest positive for each anchor
            # shape: (batch_size,)
            hardest_positive_dists = anchor_positive_dists.max(dim=1)[0]
            num_positive_pairs = torch.count_nonzero(hardest_positive_dists)
            positive_loss = hardest_positive_dists.sum() / torch.max(
                num_positive_pairs, torch.tensor(1e-16)
            )

            # get the hardest negative for each anchor
            # shape (batch_size,)
            hardest_negative_dists = anchor_negative_dists.min(dim=1)[0]
            num_negative_pairs = torch.sum(
                (
                    hardest_negative_dists
                    < max_value_of_dtype(
                        hardest_negative_dists.dtype
                    )  # It's True where we didn't set to this maximum value to mark them invalid
                ).float()
            )
            negative_loss = F.relu(
                self._margin - hardest_negative_dists
            ).sum() / torch.max(num_negative_pairs, torch.tensor(1e-16))

        total_loss = 0.5 * (positive_loss + negative_loss)

        return total_loss
