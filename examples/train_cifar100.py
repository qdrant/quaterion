import os
from typing import Dict, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from quaterion import Quaterion, TrainableModel
from quaterion.dataset import (
    GroupSimilarityDataLoader,
    SimilarityGroupSample,
    SimilarityGroupDataset,
)
from quaterion.loss import TripletLoss, SimilarityLoss
from quaterion_models.heads import EmptyHead, EncoderHead
from quaterion_models.types import CollateFnType
from quaterion_models.encoders import Encoder


try:
    import torchvision
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
except ImportError:
    import sys

    print("You need to install torchvision for this example")
    sys.exit(1)


def get_dataloader():
    # Mean and std values taken from https://github.com/LJY-HY/cifar_pytorch-lightning/blob/master/datasets/CIFAR.py#L43
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    path = os.path.join(os.path.expanduser("~"), "torchvision", "datasets")

    transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    dataset = SimilarityGroupDataset(
        datasets.CIFAR100(root=path, download=True, transform=transform)
    )
    dataloader = GroupSimilarityDataLoader(dataset, batch_size=128, shuffle=True)
    return dataloader


class MobilenetV3Encoder(Encoder):
    def __init__(self, embedding_size: int):
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

    def get_collate_fn(self) -> CollateFnType:
        return self._simple_collate_fn

    def _simple_collate_fn(self, batch):
        return torch.stack(batch)


class Model(TrainableModel):
    def __init__(self, embedding_size: int = 128, lr: float = 1e-4):
        self._embedding_size = embedding_size
        self._lr = lr
        super().__init__()

    def configure_encoders(self) -> Union[Encoder, Dict[str, Encoder]]:
        return MobilenetV3Encoder(self._embedding_size)

    def configure_head(self, input_embedding_size) -> EncoderHead:
        return EmptyHead(input_embedding_size)

    def configure_loss(self) -> SimilarityLoss:
        return TripletLoss()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), self._lr)
        return optimizer


model = Model()
train_dataloader = get_dataloader()

trainer = pl.Trainer(
    gpus=1 if torch.cuda.is_available() else 0, num_nodes=1, max_epochs=10
)

Quaterion.fit(
    trainable_model=model, trainer=trainer, train_dataloader=train_dataloader,
)
