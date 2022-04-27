import shutil

import argparse
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from quaterion import Quaterion

from .data import get_dataloaders
from .models import Model


def train(
    lr: float,
    mining: str,
    batch_size: int,
    epochs: int,
    input_size: int,
    shuffle: bool,
    save_dir: str,
):

    model = Model(
        lr=lr,
        mining=mining,
    )

    train_dataloader, val_dataloader = get_dataloaders(
        batch_size=batch_size, input_size=input_size, shuffle=shuffle
    )

    checkpoint_callback = ModelCheckpoint(
        filename="epoch{epoch}-step{step}-val_loss{validation_loss:.4f}",
        monitor="validation_loss",
        auto_insert_metric_name=False,
    )

    early_stopping = EarlyStopping(
        monitor="validation_loss",
        patience=30,
    )

    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
        max_epochs=epochs,
        callbacks=[early_stopping],  # [checkpoint_callback],
        enable_checkpointing=False,
        log_every_n_steps=1,
    )

    Quaterion.fit(
        trainable_model=model,
        trainer=trainer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
    )

    shutil.rmtree(save_dir, ignore_errors=True)
    model.save_servable(save_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=1024, help="Batch size")

    ap.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Maximum number of epochs to run training",
    )

    ap.add_argument(
        "--input-size",
        type=int,
        default=336,
        help="Images will be resized to this dimension",
    )

    ap.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    ap.add_argument(
        "--mining",
        default="hard",
        choices=("all", "hard"),
        help="Type of mining for Triplet Loss",
    )

    ap.add_argument(
        "--save-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "cars_encoders"),
        help="Where to save the servable model",
    )

    ap.add_argument(
        "--shuffle",
        type=bool,
        default=False,
        help="Whether or not to shuffle dataset during for each epoch",
    )

    args = ap.parse_args()

    train(
        args.lr,
        args.mining,
        args.batch_size,
        args.epochs,
        args.input_size,
        args.shuffle,
        args.save_dir,
    )
