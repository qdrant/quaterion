"""NOTE:
This sample script is design to fit in the memory of a 4GB GPU.
You can increase the XBM buffer size that can fit your GPU's memory
        to get most out of the XBM feature.
"""

import argparse
import os
from typing import Dict, List, Union

import pytorch_lightning as pl
import torch
from quaterion_models.encoders import Encoder
from quaterion_models.heads import EmptyHead, EncoderHead

from quaterion import Quaterion, TrainableModel
from quaterion.eval.attached_metric import AttachedMetric
from quaterion.eval.group import RetrievalRPrecision
from quaterion.loss import SimilarityLoss, TripletLoss
from quaterion.train.xbm import XbmConfig

from .cifar100 import get_dataloaders, MobilenetV3Encoder


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
        return TripletLoss(mining="semi_hard")

    def configure_xbm(self) -> XbmConfig:
        return XbmConfig(buffer_size=2048)

    def configure_metrics(self) -> Union[AttachedMetric, List[AttachedMetric]]:
        return AttachedMetric(
            name="rrp", metric=RetrievalRPrecision(), prog_bar=True, on_epoch=True
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), self._lr, weight_decay=0.0005
        )
        return optimizer


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--embedding-size", type=int, default=128, help="Size of the embedding vector"
    )

    ap.add_argument("--epochs", type=int, default=30,
                    help="Maximum number of epochs to train")

    ap.add_argument("--lr", type=float, default=1e-4, help="Learning rate")

    args = ap.parse_args()

    model = Model(
        embedding_size=args.embedding_size,
        lr=args.lr,
    )

    train_dataloader, val_dataloader = get_dataloaders(batch_size=64)

    trainer_kwargs = Quaterion.trainer_defaults(model, train_dataloader)
    trainer_kwargs['max_epochs'] = args.epochs
    trainer = pl.Trainer(**trainer_kwargs)

    Quaterion.fit(
        trainable_model=model,
        trainer=trainer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
    )
