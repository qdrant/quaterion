from typing import Type

import pytorch_lightning as pl
import torch
from quaterion_models.model import MetricModel
from torch.optim import Optimizer, Adam

from quaterion.loss.similarity_loss import SimilarityLoss


class TrainableModel(pl.LightningModule):

    def __init__(self,
                 model: MetricModel,
                 loss: SimilarityLoss,
                 optimizer_class: Type[Optimizer] = Adam,
                 optimizer_params: dict = None,
                 *args, **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self._optimizer_class = optimizer_class
        self._optimizer_params = optimizer_params or {}
        self._model = model
        self._loss = loss

    @property
    def model(self):
        return self._model

    @property
    def loss(self):
        return self._loss

    def training_step(self, batch, batch_idx, **kwargs) -> torch.Tensor:
        features, targets = batch
        embeddings = self._model(features)
        loss = self._loss(embeddings=embeddings, **targets)
        self.log("train_loss", loss)

        return loss

    def configure_optimizers(self):
        return self._optimizer_class(self._model.parameters(), **self._optimizer_params)
