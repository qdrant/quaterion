from typing import Type, Dict, Any

import torch

from torch import Tensor, LongTensor

from quaterion.distances import Distance
from quaterion.loss.pairwise_loss import PairwiseLoss
from quaterion.utils import max_value_of_dtype


class ContrastiveLoss(PairwiseLoss):
    """Contrastive loss.

    Expects as input two texts and a label of either 0 or 1. If the label == 1, then the
    distance between the two embeddings is reduced. If the label == 0, then the distance
    between the embeddings is increased.

    Further information:
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

    Args:
        distance_metric_name: Name of the function, e.g., :class:`~quaterion.distances.Distance`.
            Optional, defaults to :attr:`~quaterion.distances.Distance.COSINE`.
        margin: Negative samples (label == 0) should have a distance of at least the
            margin value.
        size_average: Average by the size of the mini-batch.
    """

    def __init__(
        self,
        distance_metric_name: Distance = Distance.COSINE,
        margin: float = 0.5,
        size_average: bool = True,
    ):
        super().__init__(distance_metric_name=distance_metric_name)
        self.margin = margin
        self.size_average = size_average

    def get_config_dict(self) -> Dict[str, Any]:
        """Config used in saving and loading purposes.

        Config object has to be JSON-serializable.

        Returns:
            Dict[str, Any]: JSON-serializable dict of params
        """
        return {
            **super().get_config_dict(),
            "margin": self.margin,
            "size_average": self.size_average,
        }

    def forward(
        self,
        embeddings: Tensor,
        pairs: LongTensor,
        labels: Tensor,
        subgroups: Tensor,
        **kwargs
    ) -> Tensor:
        """Compute loss value.

        Args:
            embeddings: Batch of embeddings, first half of embeddings are embeddings
                of first objects in pairs, second half are embeddings of second objects
                in pairs.
            pairs: Indices of corresponding objects in pairs.
            labels: Scores of positive and negative objects.
            subgroups: subgroups to distinguish objects which can and cannot be used
                as negative examples
            **kwargs: additional key-word arguments for generalization of loss call

        Returns:
            Tensor: averaged or summed loss value
        """
        rep_anchor = embeddings[pairs[:, 0]]
        rep_other = embeddings[pairs[:, 1]]
        distances = self.distance_metric.distance(rep_anchor, rep_other)
        negative_distances_impact = 0.0

        if len(subgroups.unique()) > 1:
            # shape (2 * batch_size, embeddings_size)
            embeddings_count = embeddings.shape[0]  # `embeddings_count` consists of
            # number of embeddings for `obj_a` and `obj_b`

            # `subgroups` shape is (embeddings_count,)
            # shape (embeddings_count, embeddings_count)
            subgroup_matrix: Tensor = subgroups.repeat(embeddings_count, 1)
            # shape (embeddings_count, embeddings_count)
            comp_matrix: Tensor = subgroup_matrix != subgroup_matrix.T
            # a matrix to take into account only distances to negative
            # examples, i.e. from examples which don't belong to current
            # subgroup

            # shape (embeddings_count, embeddings_count)
            distance_matrix = self.distance_metric.distance_matrix(embeddings)
            distance_matrix[~comp_matrix] = max_value_of_dtype(distance_matrix.dtype)
            # shape (embeddings_count, 1)
            negative_distances, _ = distance_matrix.min(dim=1)  # find negative examples
            # which are the closest to positive ones
            # shape (embeddings_count // 2, 1)
            neg_dist_to_anchors = negative_distances[pairs[:, 0]]
            # shape (embeddings_count // 2, 1)
            neg_dist_to_other = negative_distances[pairs[:, 1]]
            # shape (embeddings_count // 2, 1)
            negative_distances_impact = torch.relu(
                self.margin - neg_dist_to_anchors
            ).pow(2) + torch.relu(self.margin - neg_dist_to_other).pow(2)

        # shape (embeddings_count // 2, 1)
        losses = (
            0.5
            * (
                labels.float() * distances.pow(2)
                + (1 - labels).float() * torch.relu(self.margin - distances).pow(2)
            )
            + negative_distances_impact
        )

        return losses.mean() if self.size_average else losses.sum()
