import os.path

import pytorch_lightning as pl
import tempfile
import torch
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from quaterion_models.encoders import Encoder
from quaterion_models.heads import EncoderHead, GatedHead
from torch import nn
from typing import Union, Dict, Optional

from quaterion import Quaterion, TrainableModel
from quaterion.dataset import GroupSimilarityDataLoader
from quaterion.dataset.similarity_dataset import SimilarityGroupDataset
from quaterion.loss import SimilarityLoss, OnlineContrastiveLoss
from quaterion.train.cache import CacheConfig, CacheType


class MobilenetV3Encoder(Encoder):
    def save(self, output_path: str):
        pass

    @classmethod
    def load(cls, input_path: str) -> "Encoder":
        return MobilenetV3Encoder()

    def __init__(self, embedding_size=128):
        super().__init__()
        import torchvision

        self.encoder = torchvision.models.mobilenet_v3_small(pretrained=True)
        self.encoder.classifier = nn.Sequential(nn.Linear(576, embedding_size))

        self._embedding_size = embedding_size

    @property
    def trainable(self) -> bool:
        return False

    @property
    def embedding_size(self) -> int:
        return self._embedding_size

    def forward(self, images):
        return self.encoder.forward(images)


class Model(TrainableModel):
    def __init__(self, embedding_size: int, lr: float, cache_path: str):
        self._cache_path = cache_path
        self._embedding_size = embedding_size
        self._lr = lr
        super().__init__()

    def configure_encoders(self) -> Union[Encoder, Dict[str, Encoder]]:
        return MobilenetV3Encoder(self._embedding_size)

    def configure_head(self, input_embedding_size) -> EncoderHead:
        return GatedHead(input_embedding_size)

    def configure_loss(self) -> SimilarityLoss:
        return OnlineContrastiveLoss()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), self._lr)
        return optimizer

    def configure_caches(self) -> Optional[CacheConfig]:
        return CacheConfig(cache_type=CacheType.AUTO, save_dir=self._cache_path)


def test_tensor_dataset_cache():
    from torchvision import transforms
    from torchvision.datasets import FakeData

    tmp_ckpt_dir = tempfile.TemporaryDirectory()
    tmp_cache_dir = tempfile.TemporaryDirectory()

    transform = transforms.Compose(
        [
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    dataset = SimilarityGroupDataset(FakeData(size=100, transform=transform))
    dataloader = GroupSimilarityDataLoader(dataset, batch_size=4)
    val_dataloader = GroupSimilarityDataLoader(dataset, batch_size=4)

    model = Model(
        embedding_size=128,
        lr=0.001,
        cache_path=tmp_cache_dir.name,
    )

    trainer = pl.Trainer(
        callbacks=[ModelCheckpoint(dirpath=tmp_ckpt_dir.name, filename="test")],
        logger=False,
        max_epochs=1,
    )

    Quaterion.fit(
        trainable_model=model,
        trainer=trainer,
        train_dataloader=dataloader,
        val_dataloader=val_dataloader,
    )

    # Same, but with checkpoint
    print("--------- with checkpoint ----------")
    ckpt_path = os.path.join(tmp_ckpt_dir.name, "test.ckpt")

    dataset = SimilarityGroupDataset(FakeData(size=100, transform=transform))
    dataloader = GroupSimilarityDataLoader(dataset, batch_size=4)

    model = Model(
        embedding_size=128,
        lr=0.001,
        cache_path=tmp_cache_dir.name,
    )

    trainer = pl.Trainer(
        callbacks=[ModelCheckpoint(dirpath=tmp_ckpt_dir.name, filename="test")],
        logger=False,
        max_epochs=3,
    )

    Quaterion.fit(
        trainable_model=model,
        trainer=trainer,
        train_dataloader=dataloader,
        ckpt_path=ckpt_path,
    )
