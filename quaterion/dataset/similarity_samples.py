from dataclasses import dataclass
from typing import Any


@dataclass
class SimilarityPairSample:
    """
    Example:
        ```
        data = [
            # First query group (subgroup)
            SimilarityPairSample(obj_a="cheesecake", obj_b="muffins", score=0.9, subgroup=10),
            SimilarityPairSample(obj_a="cheesecake", obj_b="macaroons", score=0.8, subgroup=10),
            SimilarityPairSample(obj_a="cheesecake", obj_b="candies", score=0.7, subgroup=10),
            SimilarityPairSample(obj_a="cheesecake", obj_b="nutella", score=0.6, subgroup=10),

            # Second query group
            SimilarityPairSample(obj_a="lemon", obj_b="lime", score=0.9, subgroup=11),
            SimilarityPairSample(obj_a="lemon", obj_b="orange", score=0.7, subgroup=11),
            SimilarityPairSample(obj_a="lemon", obj_b="grapefruit", score=0.6, subgroup=11),
            SimilarityPairSample(obj_a="lemon", obj_b="mandarin", score=0.6, subgroup=11),
        ]
        ```
    """

    obj_a: Any
    obj_b: Any
    score: float
    # Consider all examples outside this group as negative samples.
    # By default, all samples belong to group 0 - therefore other samples could
    # not be used as negative examples.
    subgroup: int = 0


@dataclass
class SimilarityGroupSample:
    """
    Represent groups of similar objects all of which should match with one-another within the group.

    Example:
        Faces dataset. All pictures of a single person should have single unique group id.
        In this case NN will learn to match all pictures within the group closer to each-other, but
        pictures from different groups - further.

        ```csv
        file_name,group_id
        barak_obama_1.jpg,371
        barak_obama_2.jpg,371
        barak_obama_3.jpg,371
        elon_musk_1.jpg,555
        elon_musk_2.jpg,555
        elon_musk_3.jpg,555
        leonard_nimoy_1.jpg,209
        leonard_nimoy_2.jpg,209
        ```
    """

    obj: Any
    group: int
