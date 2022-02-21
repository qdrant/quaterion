import os.path

import pytest

import pytorch_lightning as pl
import torchvision
from quaterion_models.encoders import Encoder
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST

from quaterion import Quaterion
from quaterion.dataset import GroupSimilarityDataLoader
from quaterion.dataset.similarity_dataset import SimilarityGroupDataset


class MobilenetV3Encoder(Encoder):
    def save(self, output_path: str):
        pass

    @classmethod
    def load(cls, input_path: str) -> "Encoder":
        return MobilenetV3Encoder()

    def __init__(self, embedding_size=128):
        super().__init__()
        self.encoder = torchvision.models.mobilenet_v3_small(pretrained=True)
        self.encoder.classifier = nn.Sequential(nn.Linear(576, embedding_size))

        self._embedding_size = embedding_size

    def trainable(self) -> bool:
        return True

    def embedding_size(self) -> int:
        return self._embedding_size

    def forward(self, images):
        return self.encoder.forward(images)


@pytest.mark.skip(reason="Not yet finished")
def test_tensor_dataset_cache():
    tmp_dir_name = os.path.join(os.path.dirname(__file__), "data", "mnist")
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    dataset = SimilarityGroupDataset(
        MNIST(tmp_dir_name, download=True, transform=transform)
    )
    dataloader = GroupSimilarityDataLoader(dataset, batch_size=4)

    trainer = pl.Trainer(
        logger=False,
        max_epochs=1
    )

    Quaterion.fit(
        trainable_model=...,
        trainer=trainer,
        train_dataloader=dataloader
    )
