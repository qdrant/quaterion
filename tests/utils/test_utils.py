from typing import Iterator

import pytest

from quaterion.utils.utils import iter_by_batch
from collections.abc import Iterable
from torch.utils.data import Dataset, IterableDataset


class FakeIterableDataset(IterableDataset):
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        return iter(self.data)


class FakeDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class FakeIterable(Iterable):
    def __init__(self, container):
        self.iterator = container
        self.size = len(self.iterator)
        self.step = 0

    def __iter__(self) -> Iterator:
        return self

    def __next__(self):
        if self.step == self.size:
            raise StopIteration()

        res = self.iterator[self.step]
        self.step += 1
        return res


@pytest.mark.parametrize(
    "data, exp_batches",
    [
        # sized with __getitem__
        ([], []),
        ([1], [[1]]),
        ([1, 2], [[1, 2]]),
        ([1, 2, 3], [[1, 2], [3]]),
        ([1, 2, 3, 4], [[1, 2], [3, 4]]),
        # iterable
        (FakeIterable([]), []),
        (FakeIterable([1]), [[1]]),
        (FakeIterable([1, 2]), [[1, 2]]),
        (FakeIterable([1, 2, 3]), [[1, 2], [3]]),
        (FakeIterable([1, 2, 3, 4]), [[1, 2], [3, 4]]),
        # Dataset with __getitem__
        (FakeDataset([]), []),
        (FakeDataset([1]), [[1]]),
        (FakeDataset([1, 2]), [[1, 2]]),
        (FakeDataset([1, 2, 3]), [[1, 2], [3]]),
        (FakeDataset([1, 2, 3, 4]), [[1, 2], [3, 4]]),
        # IterableDataset
        (FakeIterableDataset([]), []),
        (FakeIterableDataset([1]), [[1]]),
        (FakeIterableDataset([1, 2]), [[1, 2]]),
        (FakeIterableDataset([1, 2, 3]), [[1, 2], [3]]),
        (FakeIterableDataset([1, 2, 3, 4]), [[1, 2], [3, 4]]),
    ],
)
def test_iter_by_batch(data, exp_batches):
    batch_size = 2
    batches = [batch for batch in iter_by_batch(data, batch_size)]
    assert batches == exp_batches
