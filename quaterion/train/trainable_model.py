from typing import Type

import pytorch_lightning as pl
import torch
from quaterion_models.model import MetricModel
from torch.optim import Optimizer, Adam

from quaterion.loss.similarity_loss import SimilarityLoss


class TrainableModel(pl.LightningModule):

    @property
    def model(self) -> MetricModel:
        raise NotImplementedError()

    @property
    def loss(self) -> SimilarityLoss:
        raise NotImplementedError()

    def process_results(self, embeddings: torch.Tensor, targets: Dict[str, Any]):
        """
        Define any additional evaluations of embeddings here.

        :param embeddings: Tensor of batch embeddings, shape: [batch_size x embedding_size]
        :param targets: Output of batch target collate
        :return: None
        """
        pass

    def training_step(self, batch, batch_idx, **kwargs) -> torch.Tensor:
        features, targets = batch
        embeddings = self.model(features)
        loss = self.loss(embeddings=embeddings, **targets)
        self.log("train_loss", loss)

        self.process_results(embeddings=embeddings, targets=targets)

        return loss

