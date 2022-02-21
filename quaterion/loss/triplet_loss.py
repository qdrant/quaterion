from typing import Optional

import torch
from quaterion.loss.group_loss import GroupLoss
from quaterion.loss.metrics import SiameseDistanceMetric


def _get_triplet_mask(labels: torch.Tensor) -> torch.Tensor:
    """Creates a 3D mask of valid triplets for the batch-all strategy.

    Given a batch of labels with `shape = (batch_size,)`
    the number of possible triplets that can be formed is:
    batch_size^3, i.e. cube of batch_size,
    which can be represented as a tensor with `shape = (batch_size, batch_size, batch_size)`.
    However, a triplet is valid if:
    `labels[i] == labels[j] and labels[i] != labels[k]`
    and `i`, `j` and `k` are distinct indices.
    This function calculates a mask indicating which ones of all the possible triplets
    are actually valid triplets based on the given criteria above.

    Args:
        labels (torch.Tensor): Labels associated with embeddings in the batch. Shape: (batch_size,)

    Returns:
        torch.Tensor: Triplet mask. Shape: (batch_size, batch_size,  batch_size)
    """
    # get a mask for distinct indices
    # Shape: (batch_size, batch_size)
    indices_equal = torch.eye(labels.size()[0], dtype=torch.bool, device=labels.device)
    indices_not_equal = torch.logical_not(indices_equal)

    # Shape: (batch_size, batch_size, 1)
    i_not_equal_j = indices_not_equal.unsqueeze(2)
    # Shape: (batch_size, 1, batch_size)
    i_not_equal_k = indices_not_equal.unsqueeze(1)
    # Shape: (1, batch_size, batch_size)
    j_not_equal_k = indices_not_equal.unsqueeze(0)

    # Shape: (batch_size, batch_size, batch_size)
    distinct_indices = torch.logical_and(
        torch.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k
    )

    # get a mask for:
    # labels[i] == labels[j] and labels[i] != labels[k]
    # Shape: (batch_size, batch_size)
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    # Shape: (batch_size, batch_size, 1)
    i_equal_j = labels_equal.unsqueeze(2)
    # Shape: (batch_size, 1, batch_size)
    i_equal_k = labels_equal.unsqueeze(1)
    # Shape: (batch_size, batch_size, batch_size)
    valid_indices = torch.logical_and(i_equal_j, torch.logical_not(i_equal_k))

    # combine masks
    mask = torch.logical_and(distinct_indices, valid_indices)

    return mask


def _get_anchor_positive_mask(labels: torch.Tensor) -> torch.Tensor:
    """Creates a 2D mask of valid anchor-positive pairs.

    Args:
        labels (torch.Tensor): Labels associated with embeddings in the batch. Shape: (batch_size,)

    Returns:
        torch.Tensor: Anchor-positive mask. Shape: (batch_size, batch_size)
    """
    # get a mask for distinct i and j indices
    # Shape: (batch_size, batch_size)
    indices_equal = torch.eye(labels.size()[0], dtype=torch.bool, device=labels.device)
    indices_not_equal = torch.logical_not(indices_equal)

    # get a mask for labels[i] == labels[j]
    # Shape: (batch_size, batch_size)
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)

    # combine masks
    mask = torch.logical_and(indices_not_equal, labels_equal)

    return mask


def _get_anchor_negative_mask(labels: torch.Tensor) -> torch.Tensor:
    """Creates a 2D mask of valid anchor-negative pairs.

    Args:
        labels (torch.Tensor): Labels associated with embeddings in the batch. Shape: (batch_size,)

    Returns:
        torch.Tensor: Anchor-negative mask. Shape: (batch_size, batch_size)
    """
    # get a mask for labels[i] != labels[k]
    # Shape: (batch_size, batch_size)
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    mask = torch.logical_not(labels_equal)

    return mask


class TripletLoss(GroupLoss):
    """Implements Triplet Loss as defined in https://arxiv.org/abs/1503.03832

    It supports batch-all and batch-hard strategies for online triplet mining.

    Args:
        margin (float, optional): Margin value to push negative examples
            apart. Defaults to `0.5`.
        squared (bool, optional): Squared Euclidean distance or not. Defaults to `True`.
        mining (str, optional): Triplet mining strategy. One of
            `"all"`, `"hard"`. Defaults to `"all"`.
    """

    def __init__(
        self,
        margin: Optional[float] = 0.5,
        squared: Optional[bool] = True,
        mining: Optional[str] = "all",
    ):
        super(TripletLoss, self).__init__()
        mining_types = ["all", "hard"]
        if mining not in mining_types:
            raise ValueError(
                f"Unrecognized mining strategy: {mining}. Must be one of {', '.join(mining_types)}"
            )

        self._margin = margin
        self._squared = squared
        self._mining = mining

    def get_config_dict(self):
        return {
            "margin": self._margin,
            "squared": self._squared,
            "mining": self._mining,
        }

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
        dists = SiameseDistanceMetric.euclidean(
            x=embeddings, matrix=True, squared=self._squared
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
            mask = _get_triplet_mask(groups).float()
            triplet_loss = mask * triplet_loss

            # get rid of easy triplets
            triplet_loss = torch.max(triplet_loss, torch.tensor(0.0))

            # get the number of triplets with a positive loss
            num_positive_triplets = torch.sum((triplet_loss > 1e-16).float())

            # get scalar loss value
            triplet_loss = torch.sum(triplet_loss) / (num_positive_triplets + 1e-16)

        else:  # batch-hard triplet mining

            # get the hardest positive for each anchor
            anchor_positive_mask = _get_anchor_positive_mask(groups).float()
            anchor_positive_dists = (
                anchor_positive_mask * dists
            )  # invalid pairs set to 0
            # Shape: (batch_size,)
            hardest_positive_dists = anchor_positive_dists.max(dim=1)[0]

            # get the hardest negative for each anchor
            anchor_negative_mask = _get_anchor_negative_mask(groups).float()
            anchor_negative_dists = dists + dists.max(dim=1, keepdim=True)[0] * (
                1.0 - anchor_negative_mask
            )  # add maximum of each row to invalid pairs
            hardest_negative_dists = anchor_negative_dists.min(dim=1)[0]

            # combine hardest positives and hardest negatives
            triplet_loss = torch.max(
                hardest_positive_dists - hardest_negative_dists + self._margin,
                torch.tensor(0.0),
            )

            # get scalar loss value
            triplet_loss = triplet_loss.mean()

        return triplet_loss
