import argparse
import os
from typing import Dict, List, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from quaterion_models.encoders import Encoder
from quaterion_models.heads import EmptyHead, EncoderHead

from quaterion import Quaterion, TrainableModel
from quaterion.dataset import GroupSimilarityDataLoader, SimilarityGroupDataset
from quaterion.eval.attached_metric import AttachedMetric
from quaterion.eval.group import RetrievalRPrecision
from quaterion.loss import SimilarityLoss, TripletLoss
from quaterion.train.xbm import XbmConfig

try:
    import torchvision
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
except ImportError:
    import sys

    print("You need to install torchvision for this example")
    sys.exit(1)


def get_dataloader():
    # Use Mean and std values for the ImageNet dataset as the base model was pretrained on it.
    # taken from https://www.geeksforgeeks.org/how-to-normalize-images-in-pytorch/
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
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
    dataloader = GroupSimilarityDataLoader(dataset, batch_size=512, shuffle=True)
    return dataloader


class MobilenetV3Encoder(Encoder):
    def __init__(self, embedding_size: int):
        super().__init__()
        self.encoder = torchvision.models.mobilenet_v3_small(pretrained=True)
        self.encoder.classifier = nn.Sequential(nn.Linear(576, embedding_size))

        self._embedding_size = embedding_size

    @property
    def trainable(self) -> bool:
        return True

    @property
    def embedding_size(self) -> int:
        return self._embedding_size

    def forward(self, images):
        return self.encoder.forward(images)


class Model(TrainableModel):
    def __init__(self, embedding_size: int, lr: float, mining: str):
        self._embedding_size = embedding_size
        self._lr = lr
        self._mining = mining
        super().__init__()

    def configure_encoders(self) -> Union[Encoder, Dict[str, Encoder]]:
        return MobilenetV3Encoder(self._embedding_size)

    def configure_head(self, input_embedding_size) -> EncoderHead:
        return EmptyHead(input_embedding_size)

    def configure_loss(self) -> SimilarityLoss:
        return TripletLoss(mining="semi_hard")

    def configure_xbm(self) -> XbmConfig:
        return XbmConfig()

    def configure_metrics(self) -> Union[AttachedMetric, List[AttachedMetric]]:
        return AttachedMetric(
            name="rrp", metric=RetrievalRPrecision(), prog_bar=True, on_epoch=True
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), self._lr, weight_decay=0.0005
        )
        return optimizer


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--embedding-size", type=int, default=128, help="Size of the embedding vector"
    )

    ap.add_argument("--lr", type=float, default=1e-4, help="Learning rate")

    ap.add_argument(
        "--mining",
        default="hard",
        choices=("hard", "semi_hard"),
        help="Type of mining for the Triplet Loss funcion",
    )

    args = ap.parse_args()

    model = Model(
        embedding_size=args.embedding_size,
        lr=args.lr,
        mining=args.mining,
    )

    train_dataloader = get_dataloader()

    trainer = pl.Trainer(accelerator="auto", devices=1, num_nodes=1, max_epochs=20)

    Quaterion.fit(
        trainable_model=model,
        trainer=trainer,
        train_dataloader=train_dataloader,
    )
