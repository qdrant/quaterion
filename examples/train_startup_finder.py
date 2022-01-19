import json
import os
from typing import Dict, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from quaterion import Quaterion, TrainableModel
from quaterion.dataset.similarity_data_loader import (
    GroupSimilarityDataLoader,
    SimilarityPairSample,
)
from quaterion.loss.arcface_loss import ArcfaceLoss
from quaterion.loss.similarity_loss import SimilarityLoss
from quaterion_models.encoder import CollateFnType, Encoder
from quaterion_models.heads.empty_head import EmptyHead, EncoderHead
from torch.utils.data import DataLoader, Dataset


class StartupDataset(Dataset):
    """A class that loads the Startups dataset.

    Download it if you haven't already: https://storage.googleapis.com/generall-shared-data/startups_demo.json
    """

    def __init__(self, path: str, max_samples=1000) -> None:
        """Initialize dataset.

        :param path: Path to `startups_demo.json` file on disk.
        :type path: str
        :param max_samples: Limit number of samples for demo purposes, defaults to 1000
        :type max_samples: int, optional
        """
        super().__init__()
        try:
            with open(path, "r", encoding="utf8") as f:
                self._data = [json.loads(next(f)) for _ in range(max_samples)]
        except IOError:
            print(f"Unable to load dataset from {path}. Have you downloaded it?")
            exit(1)

    def __getitem__(self, index: int) -> SimilarityPairSample:
        """Get an item from the dataset

        :param index: Index of the item
        :type index: int
        :return: An instance of `SimilarityPairSample` containing the name and description of the startup.
        :rtype: SimilarityPairSample
        """
        startup = self._data[index]
        return SimilarityPairSample(obj_a=startup["name"], obj_b=startup["description"])

    def __len__(self) -> int:
        return len(self._data)


dataset = StartupDataset("../../startups_demo.json")
