from dataclasses import dataclass
from typing import Any


@dataclass
class SimilarityPairSample:
    """

    Examples::

        data = [
            # First query group (subgroup)
            SimilarityPairSample(
                obj_a="cheesecake",
                obj_b="muffins",
                score=0.9,
                subgroup=10
            ),
            SimilarityPairSample(
                obj_a="cheesecake",
                obj_b="macaroons",
                score=0.8,
                subgroup=10
            ),
            SimilarityPairSample(
                obj_a="cheesecake",
                obj_b="candies",
                score=0.7,
                subgroup=10
            ),

            # Second query group (subgroup)
            SimilarityPairSample(
                obj_a="lemon",
                obj_b="lime",
                score=0.9,
                subgroup=11
            ),
            SimilarityPairSample(
                obj_a="lemon",
                obj_b="orange",
                score=0.7,
                subgroup=11
            ),
        ]

    """

    obj_a: Any
    obj_b: Any
    score: float = 1.0
    # Consider all examples outside this group as negative samples.
    # By default, all samples belong to group 0 - therefore other samples could
    # not be used as negative examples.
    subgroup: int = 0


@dataclass
class SimilarityGroupSample:
    """Represent groups of similar objects all of which should match with one-another
    within the group.

    Examples::

        Faces dataset.
        All pictures of a single person should have single unique group
        id. In this case NN will learn to match all pictures within
        the group closer to each-other, but pictures from different
        groups - further.

                     file_name  group_id
        0      elon_musk_1.jpg       555
        1      elon_musk_2.jpg       555
        2      elon_musk_3.jpg       555
        3  leonard_nimoy_1.jpg       209
        4  leonard_nimoy_2.jpg       209

    """

    obj: Any
    group: int
