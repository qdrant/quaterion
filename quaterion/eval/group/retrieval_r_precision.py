import torch
from torch import Tensor

from quaterion.eval.group import GroupMetric


class RetrievalRPrecision(GroupMetric):
    """Class for computation retrieval R-precision for group based data

    R-Precision is defined as the precision after R documents have been retrieved by the system,
    where R is also the total number of judged relevant documents for the given topic.
    Precision is defined as the portion of retrieved documents that are truly relevant to the given
    query topic.

    Args:
        distance_metric: function for distance matrix computation. Possible choice might be one of
            :class:`~quaterion.loss.metrics.SiameseDistanceMetric` methods.

    Examples:

        If there are 20 relevant documents in a corpus of 100 documents then 20 documents being
        retrieved. Only 15 of these 20 documents occurred to be relevant. Then
        retrieval R-precision is calculated as 15/20 = 0.75.

    """

    def __init__(self, distance_metric):
        super().__init__(distance_metric)

    def compute(self) -> Tensor:
        """Calculates retrieval R-precision

        Returns:
            Tensor: zero-size tensor
        """
        distance_matrix = self.calculate_distances()
        # assign max dist to obj on diag to ignore distance from obj to itself
        distance_matrix[torch.eye(distance_matrix.shape[0], dtype=torch.bool)] = (
            torch.max(distance_matrix) + 1
        )
        group_matrix = self.groups.repeat(self.groups.shape[0], 1)
        # objects with the same groups are true, others are false
        group_mask = torch.BoolTensor(group_matrix == self.groups.unsqueeze(1))
        # exclude obj
        group_mask[torch.eye(group_mask.shape[0], dtype=torch.bool)] = False
        # number of members for group which is on i-th position in groups
        relevant_numbers = group_mask.sum(dim=-1)
        nearest_to_furthest_ind = torch.argsort(
            distance_matrix, dim=-1, descending=False
        )
        sorted_by_distance = torch.gather(
            group_mask, dim=-1, index=nearest_to_furthest_ind
        )
        top_k_mask = (
            torch.arange(0, group_mask.shape[0], step=1).repeat(group_mask.shape[0], 1)
            < relevant_numbers
        )
        metric = sorted_by_distance[top_k_mask].float()
        return metric.mean()
