from typing import Tuple, Any, Sized, Iterator

import mmh3
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co, IterableDataset


class IndexingDataset(Dataset[Tuple[Any, T_co]]):
    def __init__(self, dataset: Dataset[T_co]):
        self._dataset = dataset

    def __len__(self) -> int:
        if isinstance(self._dataset, Sized):
            return len(self._dataset)
        else:
            raise NotImplementedError

    def __getitem__(self, index) -> Tuple[Any, T_co]:
        item = self._dataset.__getitem__(index=index)
        record_hash = mmh3.hash64(bytes(str(index), "utf-8"), signed=False)
        return record_hash, item


class IndexingIterableDataset(IterableDataset[Tuple[Any, T_co]]):
    def __init__(self, dataset: IterableDataset[T_co]):
        self._dataset = dataset

    def __len__(self) -> int:
        if isinstance(self._dataset, Sized):
            return len(self._dataset)
        else:
            raise NotImplementedError

    def __getitem__(self, index) -> Tuple[Any, T_co]:
        return index, self._dataset.__getitem__(index)

    def __iter__(self) -> Iterator[Tuple[Any, T_co]]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_info = (worker_info.id, worker_info.num_workers, worker_info.seed)

        for idx, item in enumerate(self._dataset):
            record_hash = mmh3.hash64(
                bytes(str((worker_info, idx)), "utf-8"), signed=False
            )
            yield record_hash, item
