from typing import Dict, Any, Union, Optional

import pytorch_lightning as pl
import torch
from quaterion_models.encoder import Encoder
from quaterion_models.heads.encoder_head import EncoderHead
from quaterion_models.model import MetricModel

from quaterion.loss.similarity_loss import SimilarityLoss
from quaterion.utils.enums import TrainStage


class TrainableModel(pl.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        encoders = self.configure_encoders()
        head = self.configure_head(MetricModel.get_encoders_output_size(encoders))

        self._model = MetricModel(encoders=encoders, head=head)
        self._loss = self.configure_loss()

    @property
    def model(self) -> MetricModel:
        return self._model

    @property
    def loss(self) -> SimilarityLoss:
        return self._loss

    def configure_loss(self) -> SimilarityLoss:
        raise NotImplementedError()

    def configure_encoders(self) -> Union[Encoder, Dict[str, Encoder]]:
        """
        Use this function to define an initial state of encoders.
        This function should be used to assign initial values for encoders
        before training as well as during the checkpoint loading.

        :return: Instance of the `Encoder` or dict of instances
        """
        raise NotImplementedError()

    def configure_head(self, input_embedding_size: int) -> EncoderHead:
        """
        Use this function to define an initial state for head layer of the model

        :param input_embedding_size: size of embeddings produced by encoders
        :return: Instance of `EncoderHead`
        """
        raise NotImplementedError()

    def process_results(
        self,
        embeddings: torch.Tensor,
        targets: Dict[str, Any],
        batch_idx,
        stage: TrainStage,
        **kwargs,
    ):
        """
        Define any additional evaluations of embeddings here.

        :param embeddings: Tensor of batch embeddings, shape: [batch_size x embedding_size]
        :param targets: Output of batch target collate
        :param batch_idx: ID of the processing batch
        :param stage: Train, validation or test stage
        :return: None
        """
        pass

    def training_step(self, batch, batch_idx, **kwargs) -> torch.Tensor:
        stage = TrainStage.TRAIN
        loss = self._common_step(
            batch=batch, batch_idx=batch_idx, stage=stage, **kwargs
        )
        return loss

    def validation_step(self, batch, batch_idx, **kwargs) -> Optional[torch.Tensor]:
        stage = TrainStage.VALIDATION
        self._common_step(batch=batch, batch_idx=batch_idx, stage=stage, **kwargs)

    def test_step(self, batch, batch_idx, **kwargs) -> Optional[torch.Tensor]:
        stage = TrainStage.TEST
        self._common_step(batch=batch, batch_idx=batch_idx, stage=stage, **kwargs)

    def _common_step(self, batch, batch_idx, stage: TrainStage, **kwargs):
        features, targets = batch
        embeddings = self.model(features)
        loss = self.loss(embeddings=embeddings, **targets)
        self.log(f"{stage}_loss", loss)
        self.process_results(
            embeddings=embeddings,
            targets=targets,
            batch_idx=batch_idx,
            stage=stage,
            **kwargs,
        )
        return loss

    def save_servable(self, path: str):
        """
        Save model, independent of Pytorch Lightning.

        :param path: where to save
        :return: None
        """
        self.model.save(path)
