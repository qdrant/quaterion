from collections import defaultdict
from typing import List, Any, Dict, Callable, Optional

from torch.utils.data import Dataset
from torch.utils.data.dataloader import T_co


from quaterion.dataset.similarity_data_loader import SimilarityDataLoader
from quaterion.train.encoders.cache_encoder import (
    KeyExtractorType,
    CacheCollateFnType,
    CacheCollateReturnType,
)


class CacheDataLoader(SimilarityDataLoader):
    """DataLoader used to provide convenient way of caching.

    Provides a way to avoid repeated calculations in the context of current
    process.

    Args:
        key_extractors: extract hashable keys for cache
        cached_encoders_collate_fns: mapping of encoders' collate functions returning
            pairs of keys and values while in caching, and only keys otherwise
        unique_objects_extractor: function to extract unique objects from batch
        dataset: dataset object to load data from
        **kwargs: key-word arguments to be passed to dataloader's `__init__`
    """

    def __init__(
        self,
        key_extractors: Dict[str, KeyExtractorType],
        cached_encoders_collate_fns: Dict[str, CacheCollateFnType],
        unique_objects_extractor: Callable[[List[Any]], List[Any]],
        dataset: Dataset[T_co],
        **kwargs
    ):
        # We use `pl.trainer.predict` to fill cache, but it recreates
        # dataloader instance internally. Trainer does it by collecting current
        # instance's `**kwargs` and creating a brand-new object of dataloader.
        # `**kwargs` contains `collate_fn` key which is duplicated with ours
        # and raises a TypeError: multiple values for keyword argument.
        kwargs.pop("collate_fn", None)
        super().__init__(dataset, collate_fn=self.cache_collate_fn, **kwargs)
        self.unique_objects_extractor = unique_objects_extractor
        self.cached_encoders_collate_fns = cached_encoders_collate_fns
        self.key_extractors = key_extractors
        self.seen_objects = defaultdict(set)

    @classmethod
    def fetch_unique_objects(cls, batch: List[T_co]) -> List[Any]:
        """Fetches unique objects across batch to avoid repeated calculations.

        Args:
            batch: batch of raw data

        Returns:
            List[Any]: list of unique objects
        """
        pass

    def cache_collate_fn(self, batch: List[T_co]) -> Dict[str, Optional[CacheCollateReturnType]]:
        """
        Collate used to cache batches without repeated calculations of
        embeddings in the current process

        Args:
            batch: batch of raw data

        Returns:
            Example:
            ```
            {
                "encoder_name": __batch_of_encoder__
            }
            ```
        """
        unique_objects: List[Any] = self.unique_objects_extractor(batch)

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
