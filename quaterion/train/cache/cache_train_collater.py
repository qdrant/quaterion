from collections import defaultdict
from typing import Dict, List, Any, Hashable

from quaterion.dataset.train_collater import TrainCollater
from quaterion_models.types import CollateFnType

from quaterion.train.cache.cache_config import KeyExtractorType
from quaterion.train.cache.cache_encoder import CacheMode


class CacheTrainCollater(TrainCollater):
    """:meta private:"""

    def __init__(
        self,
        pre_collate_fn,
        encoder_collates: Dict[str, "CollateFnType"],
        key_extractors: Dict[str, "KeyExtractorType"],
        cachable_encoders: List[str],
        mode: CacheMode,
    ):
        super().__init__(pre_collate_fn, encoder_collates)
        self.cachable_encoders = cachable_encoders
        self.mode = mode
        self.key_extractors = key_extractors
        self.seen_keys = defaultdict(set)

    def extract_keys(
        self, ids: List[int], features: List[Any], encoder_name
    ) -> List[Hashable]:
        """
        If custom `key_extractor` is specified for the encoder - use it instead of sequential
        number.

        Warnings: Do not use default `__hash__` implementation, because `torch.Tensor` does not
        work properly with it.

        Warnings: If you use default `__hash__` implementation, you have to handle multiprocessing
        issues yourself. You may need to set `fork` start method or `PYTHONHASHSEED` explicitly.
        """
        if encoder_name not in self.key_extractors:
            # Use default
            return ids

        # Use custom
        key_extractor = self.key_extractors[encoder_name]
        return [key_extractor(feature) for feature in features]

    def pre_encoder_collate(
        self, features: List[Any], ids: List[int] = None, encoder_name: str = None
    ):
        """
        Default implementation of per-encoder batch preparation, might be overridden
        """

        # Do nothing for non-cached encoders
        if encoder_name not in self.cachable_encoders:
            return features

        keys = self.extract_keys(ids=ids, features=features, encoder_name=encoder_name)

        if self.mode == CacheMode.FILL:
            # Output both - keys and features
            unseen_keys = []
            unseen_features = []
            for key, feature in zip(keys, features):
                if key in self.seen_keys[encoder_name]:
                    continue

                unseen_keys.append(key)
                unseen_features.append(feature)
                self.seen_keys[encoder_name].add(key)

            return unseen_keys, unseen_features

        if self.mode == CacheMode.TRAIN:
            # Output cached keys only
            return keys

        raise NotImplementedError(f"Cache mode {self.mode} is not implemented")
