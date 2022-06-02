import os.path

import dataclasses
import pickle
from enum import Enum
from torch.utils.data import Dataset
from torch.utils.data.dataset import IterableDataset
from typing import Sized

from quaterion.dataset.indexing_dataset import IndexingDataset, IndexingIterableDataset
from quaterion.dataset.similarity_samples import SimilarityGroupSample
from quaterion.dataset.similarity_samples import SimilaritySample, SimilarityPairSample


class LabelCacheMode(Enum):
    transparent = 0
    learn = 1
    read = 2


class LabelCacheDatasetMixin:
    @classmethod
    def _process_sample(cls, sample: SimilaritySample) -> SimilaritySample:
        """Convert read sample into cachable sample"""
        if isinstance(sample, SimilarityGroupSample):
            return dataclasses.replace(sample, obj=None)
        if isinstance(sample, SimilarityPairSample):
            return dataclasses.replace(sample, obj_a=None, obj_b=None)

    def __init__(self, *args, **kwargs):
        super(LabelCacheDatasetMixin, self).__init__(*args, **kwargs)
        self._cache = {}
        self._mode = LabelCacheMode.transparent

    @property
    def mode(self) -> LabelCacheMode:
        return self._mode

    def set_mode(self, mode: LabelCacheMode):
        self._mode = mode

    def process_item(self, index, item):
        if self._mode == LabelCacheMode.transparent:
            return index, item

        if self._mode == LabelCacheMode.read:
            return index, self._cache[index]

        if self._mode == LabelCacheMode.learn:
            self._cache[index] = self._process_sample(item)
            return index, item

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pickle.dump(self._cache, open(path, "wb"))

    def load(self, path):
        self._cache = pickle.load(open(path, "rb"))


class LabelCacheDataset(Dataset[SimilaritySample], LabelCacheDatasetMixin):
    def __init__(self, dataset: IndexingDataset):
        super().__init__()
        self._dataset = dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        hash_index, item = self._dataset.__getitem__(index)
        return self.process_item(hash_index, item)


class LabelCacheIterableDataset(
    IterableDataset[SimilaritySample], LabelCacheDatasetMixin
):
    def __init__(self, dataset: IndexingIterableDataset):
        super().__init__()
        self._dataset = dataset

    def __len__(self):
        if isinstance(self._dataset, Sized):
            return len(self._dataset)
        else:
            raise NotImplementedError()

    def __getitem__(self, index):
        hash_index, item = self._dataset.__getitem__(index)
        return self.process_item(hash_index, item)

    def __iter__(self):
        for hash_index, item in self._dataset:
            yield self.process_item(hash_index, item)
