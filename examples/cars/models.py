from typing import Dict, Optional, Union

import torch
from quaterion_models.encoders import Encoder
from quaterion_models.heads import EncoderHead, SkipConnectionHead
from torch import nn

from quaterion import TrainableModel
from quaterion.eval.attached_metric import AttachedMetric
from quaterion.eval.group import RetrievalRPrecision
from quaterion.loss import SimilarityLoss, TripletLoss
from quaterion.train.cache import CacheConfig, CacheType
from quaterion.train.xbm import XbmConfig

try:
    import torchvision
except ImportError:
    import sys

    print("You need to install torchvision for this example:")
    print("pip install torchvision")

    sys.exit(1)

from .encoders import CarsEncoder


class Model(TrainableModel):
    def __init__(self, lr: float, mining: str):
        self._lr = lr
        self._mining = mining
        super().__init__()

    def configure_encoders(self) -> Union[Encoder, Dict[str, Encoder]]:
        pre_trained_encoder = torchvision.models.resnet152(pretrained=True)
        pre_trained_encoder.fc = nn.Identity()
        return CarsEncoder(pre_trained_encoder)

    def configure_head(self, input_embedding_size) -> EncoderHead:
        return SkipConnectionHead(input_embedding_size, dropout=0.1)

    def configure_loss(self) -> SimilarityLoss:
        return TripletLoss(mining=self._mining, margin=0.5)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), self._lr)
        return optimizer

    def configure_caches(self) -> Optional[CacheConfig]:
        return CacheConfig(
            cache_type=CacheType.AUTO, save_dir="./cache_dir", batch_size=32
        )

    def configure_metrics(self) -> AttachedMetric:
        return AttachedMetric(
            "rrp",
            metric=RetrievalRPrecision(),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )
