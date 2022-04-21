from typing import Dict, Union

import torch
from quaterion import TrainableModel
from quaterion.loss import SimilarityLoss, TripletLoss
from quaterion_models.encoders import Encoder
from quaterion_models.heads import EncoderHead, WideningHead

from cars.encoders import CarsEncoder


class Model(TrainableModel):
    def __init__(self, embedding_size: int, lr: float, mining: str):
        self._embedding_size = embedding_size
        self._lr = lr
        self._mining = mining
        super().__init__()

    def configure_encoders(self) -> Union[Encoder, Dict[str, Encoder]]:
        return CarsEncoder(self._embedding_size)

    def configure_head(self, input_embedding_size) -> EncoderHead:
        return WideningHead(input_embedding_size)

    def configure_loss(self) -> SimilarityLoss:
        return TripletLoss(mining=self._mining)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), self._lr)
        return optimizer
