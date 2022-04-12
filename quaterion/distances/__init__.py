from enum import Enum

from quaterion.distances.base_distance import BaseDistance
from quaterion.distances.cosine import Cosine
from quaterion.distances.euclidean import Euclidean
from quaterion.distances.manhattan import Manhattan
from quaterion.distances.dot_product import DotProduct


class Distance(str, Enum):
    """An enumerator to pass distance metric names across the package."""

    EUCLIDEAN = "euclidean"
    COSINE = "cosine"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"

    @staticmethod
    def get_by_name(name: str) -> BaseDistance:
        """A simple utility method to get the distance class by name.

        You can pass a value from :class:`~Distance` enum or its string representation as an argument.
        """
        dists = {
            "cosine": Cosine,
            "euclidean": Euclidean,
            "manhattan": Manhattan,
            "dot_product": DotProduct,
        }

        try:
            return dists[name]
        except KeyError:
            raise ValueError(
                f"Unrecognized distance name: {name}. Must be one of {list(dists.keys())}"
            )
