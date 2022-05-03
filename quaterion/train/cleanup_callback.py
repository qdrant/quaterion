from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.trainer.states import TrainerFn

from quaterion.train.trainable_model import TrainableModel


class CleanupCallback(Callback):
    def teardown(
        self,
        trainer: "pl.Trainer",
        pl_module: TrainableModel,
        stage: Optional[str] = None,
    ) -> None:
        if stage == TrainerFn.FITTING:
            # If encoders were wrapped, unwrap them
            pl_module.unwrap_cache()

            trainer.reset_train_val_dataloaders()
            # Restore Data Loaders if they were modified for cache
            train_dataloader = trainer.train_dataloader.loaders
            pl_module.setup_dataloader(train_dataloader)

            if trainer.val_dataloaders:
                for val_loader in trainer.val_dataloaders:
                    pl_module.setup_dataloader(val_loader)
