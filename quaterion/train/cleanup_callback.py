import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.trainer.states import TrainerFn
from typing import Optional, Dict, Any

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
            val_dataloaders = (
                trainer.val_dataloaders.loaders if trainer.val_dataloaders else None
            )

            pl_module.setup_dataloader(train_dataloader)

            if val_dataloaders:
                pl_module.setup_dataloader(val_dataloaders)
