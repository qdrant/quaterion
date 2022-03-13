from typing import Optional

import torch
from quaterion.loss.group_loss import GroupLoss
from quaterion.loss.metrics import SiameseDistanceMetric
from quaterion.utils import (
    get_anchor_negative_mask,
    get_anchor_positive_mask,
    get_triplet_mask,
)


class TripletLoss(GroupLoss):
    """Implements Triplet Loss as defined in https://arxiv.org/abs/1503.03832

    It supports batch-all and batch-hard strategies for online triplet mining.

    Args:
        margin: Margin value to push negative examples
            apart. Optional, defaults to `0.5`.
        distance_metric_name: Name of the distance function. Optional, defaults to `euclidean`.
        squared (bool, optional): Squared Euclidean distance or not. Defaults to `True`.
        mining (str, optional): Triplet mining strategy. One of
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
        super(TripletLoss, self).__init__(distance_metric_name=distance_metric_name)

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
            # Calculate loss for all possible triplets first, then filter by group mask
            # Shape: (batch_size, batch_size, 1)
            anchor_positive_dists = dists.unsqueeze(2)
            # Shape: (batch_size, 1, batch_size)
            anchor_negative_dists = dists.unsqueeze(1)
            # All possible triplets: triplet_loss[anchor_id, positive_id, negative_id]
            # Shape: (batch_size, batch_size, batch_size)
            triplet_loss = anchor_positive_dists - anchor_negative_dists + self._margin

            # set invalid triplets to 0
            mask = get_triplet_mask(groups).float()
            triplet_loss = mask * triplet_loss

            # get rid of easy triplets
            triplet_loss = torch.max(triplet_loss, torch.tensor(0.0))

            # get the number of triplets with a positive loss
            num_positive_triplets = torch.sum((triplet_loss > 1e-16).float())

            # get scalar loss value
            triplet_loss = torch.sum(triplet_loss) / (num_positive_triplets + 1e-16)

        else:  # batch-hard triplet mining

            # get the hardest positive for each anchor
            anchor_positive_mask = get_anchor_positive_mask(groups).float()
            anchor_positive_dists = (
                anchor_positive_mask * dists
            )  # invalid pairs set to 0
            # Shape: (batch_size,)
            hardest_positive_dists = anchor_positive_dists.max(dim=1)[0]

            # get the hardest negative for each anchor
            anchor_negative_mask = get_anchor_negative_mask(groups).float()
            # add maximum of each row to invalid pairs to make sure not to count loss values from
            # those indices when we apply minimum function later on
            anchor_negative_dists = dists + dists.max(dim=1, keepdim=True)[0] * (
                1.0 - anchor_negative_mask
            )
            hardest_negative_dists = anchor_negative_dists.min(dim=1)[0]

            # combine hardest positives and hardest negatives
            triplet_loss = torch.max(
                # Division to maximal distance value scales target distances and prevent vector collapse
                (hardest_positive_dists - hardest_negative_dists) / hardest_positive_dists.max() + self._margin,
                torch.tensor(0.0),
            )

            # get scalar loss value
            triplet_loss = triplet_loss.mean()

        return triplet_loss
