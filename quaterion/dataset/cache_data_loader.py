from collections import defaultdict
from typing import List, Any, Generic, Dict, Callable
from torch.utils.data.dataloader import T_co
from torch.utils.data import Dataset

from quaterion.dataset.similarity_data_loader import SimilarityDataLoader
from quaterion.train.encoders.cache_encoder import KeyExtractorType, CacheCollateFnType


class CacheDataLoader(SimilarityDataLoader):
    def __init__(
        self,
        key_extractors: Dict[str, KeyExtractorType],
        cached_encoders_collate_fns: Dict[str, CacheCollateFnType],
        unique_objects_extractor: Callable[[List[Any]], List[Any]],
        dataset: Dataset[T_co],
        **kwargs
    ):
        super().__init__(dataset, collate_fn=self.cache_collate_fn, **kwargs)
        self.unique_objects_extractor = unique_objects_extractor
        self.cached_encoders_collate_fns = cached_encoders_collate_fns
        self.key_extractors = key_extractors
        self.seen_objects = defaultdict(set)

    @classmethod
    def fetch_unique_objects(cls, batch: List[Any]) -> List[Any]:
        pass

    def cache_collate_fn(self, batch: List[T_co]):
        """
        Collate used to cache batches without repeated calculations of
        embeddings in current process

        Args:
            batch:

        Returns:
            Example:
            ```
            {
                "encoder_name": __batch_of_encoder__
            }
            ```
        """
        unique_objects = self.unique_objects_extractor(batch)

        encoder_batches = defaultdict(list)
        for obj in unique_objects:
            for encoder_name, key_extractor in self.key_extractors.items():
                obj_key = key_extractor(obj)
                if obj_key not in self.seen_objects[encoder_name]:
                    # to be cached by this specific encoder
                    encoder_batches[encoder_name].append(obj)
                    self.seen_objects[encoder_name].add(obj_key)

        new_batch = {}
        for encoder_name, collate in self.cached_encoders_collate_fns.items():
            encoder_batch = encoder_batches[encoder_name]
            new_batch[encoder_name] = collate(encoder_batch) if encoder_batch else None

        return new_batch
