import torch

from dataclasses import dataclass
from typing import Any, List, Generic

from torch.utils.data import DataLoader
from torch.utils.data.dataloader import T_co

from quaterion.train.encoders.cache_encoder import CacheCollateFnType


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


class SimilarityDataloader(DataLoader, Generic[T_co]):
    LOCAL_CACHE = set()  # local process cache

    @classmethod
    def fetch_unique_objects(cls, batch: List[Any]) -> List[Any]:
        """
        Fetch unique objects from batch to avoid calculation of embeddings
        to repeated objects
        """
        raise NotImplementedError()

    def set_model_collate_fn(
        self, model_collate_fn: CacheCollateFnType,
    ):
        """
        Method used to set model's collate to retrieve keys and produce input
        for cache encoders
        """
        setattr(self, "model_collate", model_collate_fn)

    def cache_collate_fn(self, batch: List[T_co]):
        """
        Collate used to cache batches without repeated calculations of
        embeddings in current process
        """
        unique_objects = self.fetch_unique_objects(batch)
        model_collate = getattr(self, "model_collate", None)
        if not model_collate:
            raise AttributeError(
                "`set_model_collate_fn` must be called before caching"
            )
        samples = model_collate(unique_objects)

        new_keys = []
        new_objects = []
        for encoder_name in samples:
            keys, _ = samples[encoder_name]
            for ind, key in enumerate(keys):
                if key not in self.LOCAL_CACHE:
                    new_keys.append(key)
                    new_objects.append(unique_objects[ind])

        self.LOCAL_CACHE.update(new_keys)

        if len(new_keys) == len(batch):
            return samples
        elif len(new_keys) == 0:
            return {}

        return model_collate(new_objects)


class PairsSimilarityDataLoader(SimilarityDataloader[SimilarityPairSample]):
    @classmethod
    def collate_fn(cls, batch: List[SimilarityPairSample]):
        features = [record.obj_a for record in batch] + [
            record.obj_b for record in batch
        ]
        labels = {
            "pairs": torch.LongTensor(
                [[i, i + len(batch)] for i in range(len(batch))]
            ),
            "labels": torch.Tensor([record.score for record in batch]),
            "subgroups": torch.Tensor(
                [record.subgroup for record in batch] * 2
            ),
        }
        return features, labels

    @classmethod
    def fetch_unique_objects(cls, batch):
        unique_objects = []
        for sample in batch:
            if sample.obj_a not in unique_objects:
                unique_objects.append(sample.obj_a)
            if sample.obj_b not in unique_objects:
                unique_objects.append(sample.obj_b)
        return unique_objects


class GroupSimilarityDataLoader(SimilarityDataloader[SimilarityGroupSample]):
    LOCAL_CACHE = set()

    @classmethod
    def collate_fn(cls, batch: List[SimilarityGroupSample]):
        features = [record.obj for record in batch]
        labels = {
            "groups": torch.LongTensor([record.group for record in batch])
        }
        return features, labels

    @classmethod
    def fetch_unique_objects(cls, batch: List[Any]) -> List[Any]:
        unique_objects = []
        for sample in batch:
            if sample.obj not in unique_objects:
                unique_objects.append(sample.obj)
        return unique_objects
