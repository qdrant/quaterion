from typing import Type

import pytorch_lightning as pl
import torch
from quaterion_models.encoder import TensorInterchange

from torch.optim import Optimizer, Adam

from quaterion.loss.similarity_loss import SimilarityLoss
from quaterion_models.model import MetricModel


class TrainableModel(pl.LightningModule):

    def __init__(self,
                 model: MetricModel,
                 loss: SimilarityLoss,
                 optimizer_class: Type[Optimizer] = Adam,
                 optimizer_params: dict = None
                 ):
        super().__init__()
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params or {}
        self.model = model
        self.loss = loss

    def to_model_device(self, batch: TensorInterchange) -> TensorInterchange:
        pass

    def training_step(self, batch, batch_idx, **kwargs) -> torch.Tensor:
        features, targets = batch
        embeddings = self.model.forward(features)
        loss = self.loss(embeddings=embeddings, **targets)
        self.log("train_loss", loss)

        return loss

    def configure_optimizers(self):
        return self.optimizer_class(self.model.parameters(), **self.optimizer_params)
