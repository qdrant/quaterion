from enum import Enum

class Distance(Enum, str):
    """An enumerator to pass distance metric names across the package."""

    EUCLIDEAN = "euclidean"
    COSINE = "cosine"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"
