from __future__ import annotations
from typing import List, Tuple, Union, Dict, Any, Callable

from quaterion.dataset import SimilarityPairSample, SimilarityGroupSample
from quaterion_models.types import CollateFnType


class TrainCollater:
    """Functional object, that aggregates all required information for performing collate on train
    batches.

    Note:
        Should be serializable for sending among worker processes.

    Args:
        pre_collate_fn: function to split origin batch into ids, features and labels. Ids are means
            to keep track of repeatable usage of the same elements. Features are  commonly encoders
            input. Labels usually allow distinguishing positive and negative samples.
        encoder_collates: mapping of encoder name to its collate function
    """

    def __init__(
        self, pre_collate_fn: Callable, encoder_collates: Dict[str, CollateFnType]
    ):
        self.pre_collate_fn = pre_collate_fn
        self.encoder_collates = encoder_collates

    def pre_encoder_collate(
        self, features: List[Any], ids: List[int] = None, encoder_name: str = None
    ):
        """
        Default implementation of per-encoder batch preparation, might be overridden
        """
        return features

    def __call__(
        self,
        batch: List[Tuple[int, Union[SimilarityPairSample, SimilarityGroupSample]]],
    ):
        ids, features, labels = self.pre_collate_fn(batch)

        encoder_collate_result = {}
        for encoder_name, collate_fn in self.encoder_collates.items():
            encoder_features = self.pre_encoder_collate(features, ids, encoder_name)
            encoder_collate_result[encoder_name] = collate_fn(encoder_features)

        return encoder_collate_result, labels
