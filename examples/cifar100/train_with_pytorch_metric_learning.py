import argparse
from typing import Dict, Union

import pytorch_lightning as pl
import torch
from pytorch_metric_learning import losses, miners
from quaterion_models.encoders import Encoder
from quaterion_models.heads import EmptyHead, EncoderHead

from quaterion import Quaterion, TrainableModel
from quaterion.loss import SimilarityLoss
from quaterion.loss.extras import PytorchMetricLearningWrapper

from .cifar100 import MobilenetV3Encoder, get_dataloaders


class Model(TrainableModel):
    def __init__(self, embedding_size: int, lr: float):
        self._embedding_size = embedding_size
        self._lr = lr
        super().__init__()

    def configure_encoders(self) -> Union[Encoder, Dict[str, Encoder]]:
        return MobilenetV3Encoder(self._embedding_size)

    def configure_head(self, input_embedding_size) -> EncoderHead:
        return EmptyHead(input_embedding_size)

    def configure_loss(self) -> SimilarityLoss:
        loss = losses.ArcFaceLoss(embedding_size=self._embedding_size, num_classes=100)
        miner = miners.MultiSimilarityMiner()
        return PytorchMetricLearningWrapper(loss, miner)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), self._lr)
        return optimizer


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--embedding-size", type=int, default=128, help="Size of the embedding vector"
    )

    ap.add_argument("--lr", type=float, default=1e-4, help="Learning rate")

    args = ap.parse_args()

    model = Model(
        embedding_size=args.embedding_size,
        lr=args.lr,
    )

    train_dataloader, val_dataloader = get_dataloaders()

    trainer_kwargs = Quaterion.trainer_defaults(model, train_dataloader)
    trainer_kwargs["max_epochs"] = 10
    trainer = pl.Trainer(**trainer_kwargs)

    Quaterion.fit(
        trainable_model=model,
        trainer=trainer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
    )
