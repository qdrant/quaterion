from torch import nn
from typing import Dict, Union

import torch
from pytorch_metric_learning import losses, miners
from quaterion import TrainableModel
from quaterion.loss import SimilarityLoss, TripletLoss
from quaterion.loss.extras.pytorch_metric_learning_wrapper import (
    PytorchMetricLearningWrapper,
)
from quaterion_models.encoders import Encoder
from quaterion_models.heads import EncoderHead, WideningHead, SkipConnectionHead

try:
    import torchvision
except ImportError:
    import sys

    print("You need to install torchvision for this example:")
    print("pip install torchvision")

    sys.exit(1)

from .encoders import CarsEncoder


class Model(TrainableModel):
    def __init__(self, embedding_size: int, lr: float, mining: str):
        self._embedding_size = embedding_size
        self._lr = lr
        self._mining = mining
        super().__init__()

    def configure_encoders(self) -> Union[Encoder, Dict[str, Encoder]]:
        pre_trained_encoder = torchvision.models.resnet18(pretrained=True)
        pre_trained_encoder.fc = nn.Identity()
        return CarsEncoder(pre_trained_encoder)

    def configure_head(self, input_embedding_size) -> EncoderHead:
        # return WideningHead(input_embedding_size)
        return SkipConnectionHead(input_embedding_size)

    def configure_loss(self) -> SimilarityLoss:
        # return TripletLoss(mining=self._mining, margin=0.5)
        loss = losses.MultiSimilarityLoss()
        miner = miners.MultiSimilarityMiner()
        return PytorchMetricLearningWrapper(loss, miner)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), self._lr)
        return optimizer
