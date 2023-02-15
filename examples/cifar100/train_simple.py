import argparse
from typing import Dict, Union

import pytorch_lightning as pl
import torch
from quaterion_models.encoders import Encoder
from quaterion_models.heads import EmptyHead, EncoderHead

from quaterion import Quaterion, TrainableModel
from quaterion.loss import OnlineContrastiveLoss, SimilarityLoss, TripletLoss

from .cifar100 import get_dataloaders, MobilenetV3Encoder


class Model(TrainableModel):
    def __init__(self, embedding_size: int, lr: float, loss_fn: str, mining: str):
        self._embedding_size = embedding_size
        self._lr = lr
        self._loss_fn = loss_fn
        self._mining = mining
        super().__init__()

    def configure_encoders(self) -> Union[Encoder, Dict[str, Encoder]]:
        return MobilenetV3Encoder(self._embedding_size)

    def configure_head(self, input_embedding_size) -> EncoderHead:
        return EmptyHead(input_embedding_size)

    def configure_loss(self) -> SimilarityLoss:
        return (
            OnlineContrastiveLoss(mining=self._mining)
            if self._loss_fn == "contrastive"
            else TripletLoss(mining=self._mining)
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), self._lr)
        return optimizer


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--embedding-size", type=int, default=128, help="Size of the embedding vector"
    )

    ap.add_argument("--lr", type=float, default=1e-4, help="Learning rate")

    ap.add_argument(
        "--loss-fn",
        default="contrastive",
        choices=("contrastive", "triplet"),
        help="Loss function",
    )

    ap.add_argument("--epochs", type=int, default=30,
                    help="Maximum number of epochs to train")

    ap.add_argument(
        "--mining",
        default="hard",
        choices=("all", "hard"),
        help="Type of mining for the Contrastive or Triplet Loss funcions",
    )

    args = ap.parse_args()

    model = Model(
        embedding_size=args.embedding_size,
        lr=args.lr,
        loss_fn=args.loss_fn,
        mining=args.mining,
    )

    train_dataloader, val_dataloader = get_dataloaders()

    trainer_kwargs = Quaterion.trainer_defaults(model, train_dataloader)
    trainer_kwargs['max_epochs'] = args.epochs
    trainer = pl.Trainer(**trainer_kwargs)

    Quaterion.fit(
        trainable_model=model,
        trainer=trainer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
    )
