import random
from typing import Tuple, Any, Sized, Iterator

import mmh3
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co, IterableDataset


def _hashit(obj: Any, salt):
    return mmh3.hash64(bytes(str(obj) + str(salt), "utf-8"), signed=False)[0]


class IndexingDataset(Dataset[Tuple[Any, T_co]]):
    def __init__(self, dataset: Dataset[T_co], salt=None):
        self._dataset = dataset
        if salt is None:
            self.salt = random.randint(0, 2**31)
        else:
            self.salt = salt

        # If item is already cached - it might be much faster to just return an id without items
        self._skip_read = False

    def __len__(self) -> int:
        if isinstance(self._dataset, Sized):
            return len(self._dataset)
        else:
            raise NotImplementedError()

    def __getitem__(self, index) -> Tuple[Any, T_co]:
        if self._skip_read:
            item = None
        else:
            item = self._dataset.__getitem__(index)
        hashed_index = _hashit(index, self.salt)
        return hashed_index, item

    def set_salt(self, salt):
        self.salt = salt

    def set_skip_read(self, skip: bool):
        self._skip_read = skip


class IndexingIterableDataset(IterableDataset[Tuple[Any, T_co]]):
    def __init__(self, dataset: IterableDataset[T_co], salt=None):
        self._dataset = dataset
        if salt is None:
            self.salt = random.randint(0, 2**31)
        else:
            self.salt = salt

        # If item is already cached - it might be much faster to just return an id without items
        self._skip_read = False

    def __len__(self) -> int:
        if isinstance(self._dataset, Sized):
            return len(self._dataset)
        else:
            raise NotImplementedError()

    def __getitem__(self, index) -> Tuple[Any, T_co]:
        hashed_index = _hashit(index, self.salt)
        return hashed_index, self._dataset.__getitem__(index)

    def __iter__(self) -> Iterator[Tuple[Any, T_co]]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_info = (worker_info.id, worker_info.num_workers, worker_info.salt)

        if self._skip_read and isinstance(self._dataset, Sized):
            for idx in range(len(self._dataset)):
                record_hash = _hashit((worker_info, idx), self.salt)
                yield record_hash, None
        else:
            for idx, item in enumerate(self._dataset):
                record_hash = _hashit((worker_info, idx), self.salt)
                yield record_hash, item

    def set_salt(self, salt):
        self.salt = salt

    def set_skip_read(self, skip: bool):
        self._skip_read = skip
