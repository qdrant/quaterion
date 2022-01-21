import torch
from torch import Tensor, pairwise_distance, cosine_similarity


class SiameseDistanceMetric:
    """
    The metric for the contrastive loss
    """

    @staticmethod
    def euclidean(x: Tensor, y: Tensor) -> Tensor:
        return pairwise_distance(x, y, p=2)

    @staticmethod
    def manhattan(x: Tensor, y: Tensor) -> Tensor:
        return pairwise_distance(x, y, p=1)

    @staticmethod
    def cosine_distance(x: Tensor, y: Tensor) -> Tensor:
        return 1 - cosine_similarity(x, y)

    @staticmethod
    def dot_product_distance(x: Tensor, y: Tensor) -> Tensor:
        return -torch.einsum("id,jd->j", x, y)
