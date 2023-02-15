import os

import torch.nn as nn
from quaterion_models.encoders import Encoder
from quaterion.dataset import GroupSimilarityDataLoader, SimilarityGroupDataset

try:
    import torchvision
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
except ImportError:
    import sys

    print("You need to install torchvision for this example")
    sys.exit(1)


def get_dataloaders(batch_size: int = 128):
    # Use Mean and std values for the ImageNet dataset as the base model was pretrained on it.
    # taken from https://www.geeksforgeeks.org/how-to-normalize-images-in-pytorch/
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    path = os.path.join(os.path.expanduser("~"), "torchvision", "datasets")

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_dataset = SimilarityGroupDataset(
        datasets.CIFAR100(root=path, download=True,
                          train=True, transform=train_transform)
    )
    train_dataloader = GroupSimilarityDataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = SimilarityGroupDataset(
        datasets.CIFAR100(root=path, download=True,
                          train=False, transform=val_transform)
    )
    val_dataloader = GroupSimilarityDataLoader(
        val_dataset, batch_size=256, shuffle=False)

    return train_dataloader, val_dataloader


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
