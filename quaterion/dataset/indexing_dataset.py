import random
from typing import Tuple, Any, Sized, Iterator

import mmh3
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co, IterableDataset


def _hashit(obj: Any, seed):
    return mmh3.hash64(bytes(str(obj) + str(seed), "utf-8"), signed=False)[0]


class IndexingDataset(Dataset[Tuple[Any, T_co]]):
    def __init__(self, dataset: Dataset[T_co], seed=None):
        self._dataset = dataset
        if seed is None:
            self.seed = random.randint(0, 2**31)
        else:
            self.seed = seed

    def __len__(self) -> int:
        if isinstance(self._dataset, Sized):
            return len(self._dataset)
        else:
            raise NotImplementedError()

    def __getitem__(self, index) -> Tuple[Any, T_co]:
        item = self._dataset.__getitem__(index=index)
        record_hash = _hashit(index, self.seed)
        return record_hash, item


class IndexingIterableDataset(IterableDataset[Tuple[Any, T_co]]):
    def __init__(self, dataset: IterableDataset[T_co], seed=None):
        self._dataset = dataset
        if seed is None:
            self.seed = random.randint(0, 2**31)
        else:
            self.seed = seed

    def __len__(self) -> int:
        if isinstance(self._dataset, Sized):
            return len(self._dataset)
        else:
            raise NotImplementedError()

    def __getitem__(self, index) -> Tuple[Any, T_co]:
        return _hashit(index, self.seed), self._dataset.__getitem__(index)

    def __iter__(self) -> Iterator[Tuple[Any, T_co]]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_info = (worker_info.id, worker_info.num_workers, worker_info.seed)

        for idx, item in enumerate(self._dataset):
            record_hash = _hashit((worker_info, idx), self.seed)
            yield record_hash, item
