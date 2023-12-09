from typing import Optional

import torch
import torch.nn.functional as F
from torch import LongTensor, Tensor

from quaterion.distances import Distance
from quaterion.loss.group_loss import GroupLoss
from quaterion.utils import get_centroids, get_triplet_mask

class CentroidTripletLoss(GroupLoss):
    """Implements Centroid Triplet Loss.

    Args:
        margin: Margin value to push negative centroids apart from positive centroids.
        distance_metric_name: Name of the distance function, e.g., :class:`~quaterion.distances.Distance`.
    """

    def __init__(
        self,
        margin: Optional[float] = 0.5,
        distance_metric_name: Optional[Distance] = Distance.COSINE,
    ):
        super(CentroidTripletLoss, self).__init__(distance_metric_name=distance_metric_name)
        self._margin = margin

    def get_config_dict(self):
        return {"margin": self._margin}

    def forward(
        self,
        embeddings: Tensor,
        groups: LongTensor,
    ) -> Tensor:
        """Calculates Centroid Triplet Loss with specified embeddings and labels.

        Args:
            embeddings: shape: (batch_size, vector_length) - Batch of embeddings.
            groups: shape: (batch_size,) - Batch of labels associated with `embeddings`

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Compute centroids for each group in the batch
        centroids = get_centroids(embeddings, groups)

        # Compute distance matrix between centroids
        centroid_dists = self.distance_metric.distance_matrix(centroids)

        # Generate triplet mask for centroids (indicating valid triplets)
        triplet_mask = get_triplet_mask(groups)

        # Apply triplet mask to distance matrix
        masked_dists = centroid_dists * triplet_mask.float()

        # Compute the triplet loss using centroids and masked distances
        triplet_loss = 0.0
        
        # Find the hardest positive and negative distances for each anchor centroid
        for i in range(len(centroids)):
            anchor_centroid = centroids[i]

            # Mask distances for the current anchor
            masked_anchor_dists = masked_dists[i]

            # Positive distances (from the same group as the anchor)
            positive_dists = masked_anchor_dists[groups == groups[i]]

            # Negative distances (from different groups than the anchor)
            negative_dists = masked_anchor_dists[groups != groups[i]]

            # Find the hardest positive and negative distances
            hardest_positive_dist = torch.max(positive_dists)
            hardest_negative_dist = torch.min(negative_dists)

            # Compute the triplet loss for the current anchor
            triplet_loss += torch.max(hardest_positive_dist - hardest_negative_dist + self._margin, torch.tensor(0.0))

        # Compute the mean triplet loss across the batch
        triplet_loss /= len(centroids)

        return triplet_loss
