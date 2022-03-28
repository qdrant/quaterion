import torch

from quaterion.eval.base_metric import BaseMetric


class PairMetric(BaseMetric):
    def compute(self):
        raise NotImplementedError()

    def precompute(self):
        pairs = self.labels["pairs"]
        labels = self.labels["labels"]
        distance_matrix = self.calculate_distances()
        target = torch.zeros_like(distance_matrix)
        # todo: subgroups should also be taken into account
        target[pairs[:, 0], pairs[:, 1]] = labels
        target[pairs[:, 1], pairs[:, 0]] = labels
        return distance_matrix, target


class RetrievalReciprocalRank(PairMetric):
    def __init__(self, encoder, distance_metric):
        super().__init__(encoder, distance_metric)

    def compute(self):
        distance_matrix, target = self.precompute()
        distance_matrix[torch.eye(distance_matrix.shape[0], dtype=torch.bool)] = (
            torch.max(distance_matrix) + 1
        )

        indices = torch.argsort(distance_matrix, dim=1)
        target = target.gather(1, indices)
        position = torch.nonzero(target)
        metric = 1.0 / (position[:, 1] + 1.0)
        return metric


class RetrievalPrecision(PairMetric):
    def __init__(self, encoder, distance_metric, k=1):
        super().__init__(encoder, distance_metric)
        self.k = k
        if self.k < 1:
            raise ValueError("k must be greater than 0")

    def compute(self):
        distance_matrix, target = self.precompute()
        # assign max dist to obj on diag to ignore distance from obj to itself
        distance_matrix[torch.eye(distance_matrix.shape[0], dtype=torch.bool)] = (
            torch.max(distance_matrix) + 1
        )
        metric = (
            target.gather(1, distance_matrix.topk(self.k, dim=1, largest=False)[1])
            .sum(dim=1)
            .float()
        ) / self.k
        return metric
