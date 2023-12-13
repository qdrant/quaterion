from typing import Optional
import torch
from torch import Tensor
from quaterion.distances.base_distance import BaseDistance

class Minkowski(BaseDistance):
    """Compute Minkowski distances (and its interpretation as similarities)."""

    @staticmethod
    def distance(x: Tensor, y: Tensor, p: int = 2) -> Tensor:
        return torch.pow(torch.sum(torch.pow(torch.abs(x - y), p), dim=-1), 1/p)

    @staticmethod
    def similarity(x: Tensor, y: Tensor, p: int = 2) -> Tensor:
        return 1 / (1 + Minkowski.distance(x, y, p))

    @staticmethod
    def distance_matrix(x: Tensor, y: Optional[Tensor] = None, p: int = 2) -> Tensor:
        if y is None:
            y = x
        return torch.cdist(x, y, p=p)

    @staticmethod
    def similarity_matrix(x: Tensor, y: Optional[Tensor] = None, p: int = 2) -> Tensor:
        return 1 / (1 + Minkowski.distance_matrix(x, y, p))

# Example usage:
tensor_x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
tensor_y = torch.tensor([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])

# Compute Minkowski distance between tensors x and y
minkowski_dist = Minkowski.distance(tensor_x, tensor_y, p=3)
print("Minkowski Distance:")
print(minkowski_dist)

# Compute Minkowski similarity between tensors x and y
minkowski_sim = Minkowski.similarity(tensor_x, tensor_y, p=3)
print("\nMinkowski Similarity:")
print(minkowski_sim)

# Compute Minkowski distance matrix between tensor x and itself
minkowski_dist_matrix = Minkowski.distance_matrix(tensor_x, p=3)
print("\nMinkowski Distance Matrix:")
print(minkowski_dist_matrix)

# Compute Minkowski similarity matrix between tensor x and itself
minkowski_sim_matrix = Minkowski.similarity_matrix(tensor_x, p=3)
print("\nMinkowski Similarity Matrix:")
print(minkowski_sim_matrix)
