from typing import Optional

import torch
import torch.nn.functional as F
from torch import LongTensor, Tensor

from quaterion.distances import Distance
from quaterion.loss.group_loss import GroupLoss
from quaterion.utils import (
    get_anchor_positive_mask,
    get_masked_maximum,
    get_masked_minimum,
    get_triplet_mask,
    max_value_of_dtype,
)
from quaterion.utils.utils import get_anchor_negative_mask


class TripletLoss(GroupLoss):
    """Implements Triplet Loss as defined in https://arxiv.org/abs/1503.03832

    It supports batch-all and batch-hard strategies for online triplet mining.

    Args:
        margin: Margin value to push negative examples
            apart. Optional, defaults to `0.5`.
        distance_metric_name: Name of the distance function, e.g.,
            :class:`~quaterion.distances.Distance`. Optional, defaults to
            :attr:`~quaterion.distances.Distance.COSINE`.
        mining (str, optional): Triplet mining strategy. One of
            `"all"`, `"hard"`. Defaults to `"hard"`.
    """

    def __init__(
        self,
        margin: Optional[float] = 1.0,
        distance_metric_name: Distance = Distance.COSINE,
        mining: Optional[str] = "hard",
    ):
        mining_types = ["all", "hard", "semi_hard"]
        if mining not in mining_types:
            raise ValueError(
                f"Unrecognized mining strategy: {mining}. Must be one of {', '.join(mining_types)}"
            )
        super(TripletLoss, self).__init__(distance_metric_name=distance_metric_name)

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

    def _hard_triplet_loss(
        self,
        embeddings_a: Tensor,
        groups_a: LongTensor,
        embeddings_b: Tensor,
        groups_b: LongTensor,
    ) -> Tensor:
        """
        Calculates Triplet Loss with hard mining between two sets of embeddings.

        Args:
            embeddings_a: (batch_size_a, vector_length) - Batch of embeddings.
            groups_a: (batch_size_a,) - Batch of labels associated with `embeddings_a`
            embeddings_b: (batch_size_b, vector_length) - Batch of embeddings.
            groups_b: (batch_size_b,) - Batch of labels associated with `embeddings_b`

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Shape: (batch_size_a, batch_size_b)
        dists = self.distance_metric.distance_matrix(embeddings_a, embeddings_b)
        # get the hardest positive for each anchor
        anchor_positive_mask = get_anchor_positive_mask(groups_a, groups_b).float()
        anchor_positive_dists = anchor_positive_mask * dists  # invalid pairs set to 0
        # Shape: (batch_size,)
        hardest_positive_dists = anchor_positive_dists.max(dim=1)[0]

        # get the hardest negative for each anchor
        anchor_negative_mask = get_anchor_negative_mask(groups_a, groups_b).float()
        # add maximum of each row to invalid pairs to make sure not to count loss values from
        # those indices when we apply minimum function later on
        anchor_negative_dists = dists + dists.max(dim=1, keepdim=True)[0] * (
            1.0 - anchor_negative_mask
        )
        hardest_negative_dists = anchor_negative_dists.min(dim=1)[0]

        # combine hardest positives and hardest negatives
        triplet_loss = F.relu(
            # Division by the minimal distance between negative samples scales target distances
            # # and prevents vector collapse
            (hardest_positive_dists - hardest_negative_dists)
            / hardest_negative_dists.mean()
            + self._margin
        )

        # get scalar loss value
        triplet_loss = triplet_loss.mean()

        return triplet_loss

    def _semi_hard_triplet_loss(
        self,
        embeddings_a: Tensor,
        groups_a: Tensor,
        embeddings_b: Tensor,
        groups_b: Tensor,
    ) -> Tensor:
        """Compute triplet loss with semi-hard mining as described in https://arxiv.org/abs/1703.07737

        It encourages the positive distances to be smaller than the minimum negative distance
        among which are at least greater than the positive distance plus the margin
        (called semi-hard negative),
        i.e., D(a, p) < D(a, n) < D(a, p) + margin.
            If no such negative exists, it uses the largest negative distance instead.

            Inspired by https://github.com/tensorflow/addons/blob/master/tensorflow_addons/losses/triplet.py

        Args:
            embeddings_a: shape: (batch_size_a, vector_length) - Output embeddings from the
                encoder.
            groups_a: shape: (batch_size_a,) - Group ids associated with embeddings.
            embeddings: shape: (batch_size_b, vector_length) - Batch of bmbeddings
            groups_b: shape: (batch_size_b,) - Groups ids associated with `embeddings_b`

        Returns:
            Tensor: zero-size tensor, XBM loss value.
        """
        # compute the distance matrix
        # shape: (batch_size_a, batch_size_b)
        dists = self.distance_metric.distance_matrix(embeddings_a, embeddings_b)

        # compute masks to express the positive and negative pairs
        # shape: (batch_size_a, batch_size_b)
        groups_a = groups_a.unsqueeze(1)
        anchor_positive_pairs = groups_a == groups_b.unsqueeze(1).t()
        anchor_negative_pairs = ~anchor_positive_pairs

        batch_size = torch.numel(groups_a)

        # compute the mask to express the semi-hard-negatives
        # WARNING: `torch.repeat()` copies the underlying data
        # so it consumes more memory
        dists_tile = dists.repeat([batch_size, 1])
        mask = anchor_negative_pairs.repeat([batch_size, 1]) & (
            dists_tile > torch.reshape(dists.t(), [-1, 1])
        )

        mask_final = torch.reshape(
            torch.sum(mask, 1, keepdims=True) > 0.0, [batch_size, batch_size]
        )
        mask_final = mask_final.t()

        # negatives_outside: smallest D(a, n) where D(a, n) > D(a, p).
        negatives_outside = torch.reshape(
            get_masked_minimum(dists_tile, mask), [batch_size, batch_size]
        )
        negatives_outside = negatives_outside.t()

        # negatives_inside: largest D(a, n).
        negatives_inside = get_masked_maximum(dists, anchor_negative_pairs)
        negatives_inside = negatives_inside.repeat([1, batch_size])

        # select either semi-hard negative or the largest negative
        # based on the condition the mask previously computed
        semi_hard_negatives = torch.where(
            mask_final, negatives_outside, negatives_inside
        )

        loss_matrix = (dists - semi_hard_negatives) + self._margin

        # the paper takes all the positives accept the diagonal
        mask_positives = anchor_positive_pairs.float() - torch.eye(
            batch_size, device=groups_a.device
        )

        # average by the number of positives
        num_positives = torch.sum(mask_positives)

        triplet_loss = (
            torch.sum(
                torch.max(
                    loss_matrix * mask_positives,
                    torch.tensor([0.0], device=groups_a.device),
                )
            )
            / num_positives
        )

        return triplet_loss

    def forward(
        self,
        embeddings: Tensor,
        groups: LongTensor,
    ) -> Tensor:
        """Calculates Triplet Loss with specified embeddings and labels.

        Args:
            embeddings: shape: (batch_size, vector_length) - Batch of embeddings.
            groups: shape: (batch_size,) - Batch of labels associated with `embeddings`

        Returns:
            torch.Tensor: Scalar loss value.
        """

        if self._mining == "all":
            # Shape: (batch_size, batch_size)
            dists = self.distance_metric.distance_matrix(embeddings)

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
            triplet_loss = F.relu(triplet_loss)

            # get the number of triplets with a positive loss
            num_positive_triplets = torch.sum((triplet_loss > 1e-16).float())

            # get scalar loss value
            triplet_loss = torch.sum(triplet_loss) / (num_positive_triplets + 1e-16)

        elif self._mining == "hard":
            triplet_loss = self._hard_triplet_loss(
                embeddings, groups, embeddings, groups
            )
        else:  # semi-hard triplets
            triplet_loss = self._semi_hard_triplet_loss(
                embeddings, groups, embeddings, groups
            )

        return triplet_loss

    def xbm_loss(
        self,
        embeddings: Tensor,
        groups: LongTensor,
        memory_embeddings: Tensor,
        memory_groups: LongTensor,
    ) -> Tensor:
        """Implement XBM loss computation for this loss.

        Args:
            embeddings: shape: (batch_size, vector_length) - Output embeddings from the
                encoder.
            groups: shape: (batch_size,) - Group ids associated with embeddings.
            memory_embeddings: shape: (memory_buffer_size, vector_length) - Embeddings stored
                in a ring buffer
            memory_groups: shape: (memory_buffer_size,) - Groups ids associated with `memory_embeddings`

        Returns:
            Tensor: zero-size tensor, XBM loss value.
        """
        if len(memory_groups) == 0 or self._mining == "all":
            return torch.tensor(
                0, device=embeddings.device
            )  # no XBM loss if memory is empty or all triplets strategy is chosen

        return (
            self._hard_triplet_loss(
                embeddings, groups, memory_embeddings, memory_groups
            )
            if self._mining == "hard"
            else self._semi_hard_triplet_loss(
                embeddings, groups, memory_embeddings, memory_groups
            )
        )
