from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple, Union

from quaterion_models.types import CollateFnType, MetaExtractorFnType
from quaterion_models.utils.meta import merge_meta

from quaterion.dataset import SimilarityGroupSample, SimilarityPairSample


class TrainCollator:
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
        self,
        pre_collate_fn: Callable,
        encoder_collates: Dict[str, CollateFnType],
        meta_extractors: Dict[str, MetaExtractorFnType],
    ):
        self.pre_collate_fn = pre_collate_fn
        self.encoder_collates = encoder_collates
        self.meta_extractors = meta_extractors

    def pre_encoder_collate(
        self, features: List[Any], ids: List[int] = None, encoder_name: str = None
    ):
        """
        Default implementation of per-encoder batch preparation, might be overridden
        """
        return features

    def process_meta(self, meta: Dict[str, List]) -> Any:
        return merge_meta(meta)

    def __call__(
        self,
        batch: List[Tuple[int, Union[SimilarityPairSample, SimilarityGroupSample]]],
    ):
        ids, features, labels = self.pre_collate_fn(batch)

        encoder_collate_result = {}
        meta = {}
        for encoder_name, collate_fn in self.encoder_collates.items():
            encoder_features = self.pre_encoder_collate(features, ids, encoder_name)
            encoder_collate_result[encoder_name] = collate_fn(encoder_features)
            meta[encoder_name] = self.meta_extractors[encoder_name](encoder_features)

        return {"data": encoder_collate_result, "meta": self.process_meta(meta)}, labels
