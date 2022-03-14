from typing import Sized

from torch.utils.data import Dataset

from quaterion.dataset import SimilarityGroupSample


class SimilarityGroupDataset(Dataset[SimilarityGroupSample]):
    """Simple wrapper class, which converts standard dataset of classification task into dataset,
    compatible with :class:`~quaterion.dataset.similarity_data_loader.GroupSimilarityDataLoader`.

    Args:
        dataset: a dataset, which return data in format: `(record, label)`

    """

    def __init__(self, dataset: Dataset):
        self._dataset = dataset

    def __len__(self) -> int:
        if isinstance(self._dataset, Sized):
            return len(self._dataset)
        else:
            raise NotImplementedError

    def __getitem__(self, index) -> SimilarityGroupSample:
        record, label = self._dataset.__getitem__(index=index)
        return SimilarityGroupSample(obj=record, group=label)
