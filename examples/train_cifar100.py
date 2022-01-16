import os

import torch
from torch.utils.data import (Dataset, DataLoader)
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from quaterion import (Quaterion, TrainableModel)
from quaterion.loss.arcface_loss import ArcfaceLoss
from quaterion.loss.softmax_loss import SoftmaxLoss
from quaterion.dataset.similarity_data_loader import (
    SimilarityGroupSample, GroupSimilarityDataLoader)
from quaterion_models.encoder import Encoder
from quaterion_models.heads.empty_head import EmptyHead

try:
    import torchvision
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
except ImportError:
    import sys
    print("You need to install torchvision for this example")
    sys.exit(1)


class CIFAR100Dataset(Dataset):
    def __init__(self, train=True):
        super().__init__()
        self.mean = [0.4914, 0.4822, 0.4465]
        self.std = [0.2023, 0.1994, 0.2010]
        self.path = os.path.join(os.path.expanduser(
            '~'), 'torchvision', 'datasets')

        if train:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])

        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])

        self.data = datasets.CIFAR100(
            root=self.path, train=train, download=True, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index]
        return SimilarityGroupSample(obj=image, group=label)

    def __len__(self):
        return len(self.data)


class MobilenetV3Encoder(Encoder):
    def __init__(self):
        super().__init__()
        self.encoder = torchvision.models.mobilenet_v3_small(pretrained=True)
        self.encoder.classifier = nn.Sequential(
            nn.Linear(576, 256)
        )

    def trainable(self):
        return True

    def embedding_size(self):
        return 256

    def get_collate_fn(self):
        return lambda batch: ({'default': torch.stack([item.obj for item in batch])}, {'groups': torch.LongTensor([item.group for item in batch])})

    def forward(self, images):
        return self.encoder.forward(images)


class Model(TrainableModel):
    def __init__(self, num_classes=100, lr=1e-5):
        self.num_classes = num_classes
        self.lr = lr
        super().__init__()

    def configure_encoders(self):
        return MobilenetV3Encoder()

    def configure_head(self, input_embedding_size):
        return EmptyHead(input_embedding_size)

    def configure_loss(self):
        return ArcfaceLoss(256, self.num_classes)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.model.parameters(), 'lr': self.lr},
            {'params': self.loss.parameters(), 'lr': self.lr * 10.}
        ])
        return optimizer

    def _common_step(self, batch, batch_idx, stage, **kwargs):
        batch = batch['default']
        return super()._common_step(batch, batch_idx, stage, **kwargs)


model = Model()
train_dataloader = GroupSimilarityDataLoader(
    CIFAR100Dataset(), batch_size=128, shuffle=True)
trainer = pl.Trainer(gpus=1, num_nodes=1, max_epochs=15)

Quaterion.fit(
    trainable_model=model,
    trainer=trainer,
    train_dataloader=train_dataloader
)
