import torch

from quaterion.eval.base_metric import BaseMetric


class RetrievalRPrecision(BaseMetric):
    def __init__(self, encoder, distance_metric):
        super().__init__(encoder, distance_metric)

    def compute(self):
        groups = self.labels["groups"]
        distance_matrix = self.calculate_distances()
        # assign max dist to obj on diag to ignore distance from obj to itself
        distance_matrix[torch.eye(distance_matrix.shape[0], dtype=torch.bool)] = (
            torch.max(distance_matrix) + 1
        )
        group_matrix = groups.repeat(groups.shape[0], 1)
        # objects with the same groups are true, others are false
        group_mask = torch.BoolTensor(group_matrix == groups.unsqueeze(1))
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
