from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from quaterion.loss.group_loss import GroupLoss


def _get_distance_matrix(
    embeddings: torch.Tensor, squared: Optional[bool] = True
) -> torch.Tensor:
    """Calculates pairwise Euklidean distances between all the embeddings.

    Args:
        embeddings (torch.Tensor): Batch of embeddings. Shape: (batch_size, embedding_dim)
        squared (bool, optional): Squared Euclidean distance or not. Defaults to True.

    Returns:
        torch.Tensor: Calculated distance matrix. Shape: (batch_size, batch_size)
    """
    # Calculate dot product. Shape: (batch_size, batch_size)
    dot_product = torch.mm(embeddings, embeddings.transpose(0, 1))
    # get L2 norm by diagonal. Shape: (batch_size,)
    square_norm = torch.diagonal(dot_product)
    # calculate distances. Shape: (batch_size, batch_size)
    distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
    # get rid of negative distances due to calculation errors
    distances = torch.maximum(distances, torch.tensor(0.0))

    if not squared:
        # handle numerical stability
        mask = (distances == 0.0).float()
        distances = distances + mask * 1e-16

        distances = torch.sqrt(distances)

        # Undo trick for numerical stability
        distances = distances * (1.0 - mask)

    return distances


def _get_triplet_mask(labels: torch.Tensor) -> torch.Tensor:
    """_Create a 3D mask to filter valid triplets for the batch.

    A triplet is valid if:
    `labels[i] == labels[j] and labels[i] != labels[k]`
    and `i`, `j` and `k` are distinct.

    Args:
        labels (torch.Tensor): Labels associated with embeddings in the batch. Shape: (batch_size,)

    Returns:
        torch.Tensor: Triplet mask. Shape: (batch_size, batch_size,  batch_size)
    """
    # get a mask for distinct indices
    # Shape: (batch_size, batch_size)
    indices_equal = torch.eye(labels.size()[0]).bool().to(labels.device)
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


class TripletLoss(GroupLoss):
    """Triplet Loss as defined in https://arxiv.org/abs/1503.03832

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
        if self._mining == "all":
            # Shape: (batch_size, batch_size)
            dists = _get_distance_matrix(embeddings, squared=self._squared)
            # Shape: (batch_size, batch_size, 1)
            anchor_positive_dists = dists.unsqueeze(2)
            # Shape: (batch_size, 1, batch_size)
            anchor_negative_dists = dists.unsqueeze(1)
            # Shape: (batch_size, batch_size, batch_size)
            triplet_loss = anchor_positive_dists - anchor_negative_dists + self._margin

            # set invalid triplets to 0
            mask = _get_triplet_mask(groups).float()
            triplet_loss = mask * triplet_loss

            # get rid of easy triplets
            triplet_loss = torch.maximum(triplet_loss, torch.tensor(0.0))

            # get the number of triplets with a positive loss
            num_positive_triplets = torch.sum((triplet_loss > 1e-16).float())

            # get scalar loss value
            triplet_loss = torch.sum(triplet_loss) / (num_positive_triplets + 1e-16)

            return triplet_loss

        else:
            raise ValueError(
                "batch-hard strategy has yet to be implemented. Use mining = 'all' for now"
            )
