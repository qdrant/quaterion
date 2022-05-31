from typing import Optional, Dict, Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress.rich_progress import (
    RichProgressBar,
    RichProgressBarTheme,
)

from quaterion.train.cache.cache_model import CacheModel


class QuaterionProgressBar(RichProgressBar):
    def __init__(
        self,
        refresh_rate: int = 1,
        leave: bool = False,
        theme: RichProgressBarTheme = RichProgressBarTheme(),
        console_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            refresh_rate=refresh_rate,
            leave=leave,
            theme=theme,
            console_kwargs=console_kwargs,
        )
        self.predict_progress_bar_id = None
        self._caching = False

    def on_predict_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if not self.has_dataloader_changed(dataloader_idx):
            return

        if isinstance(pl_module, CacheModel):
            self._caching = True

        if self.predict_progress_bar_id is not None:
            self.progress.update(self.predict_progress_bar_id, advance=0, visible=False)
        self.predict_progress_bar_id = self._add_task(
            self.total_predict_batches_current_dataloader, self.predict_description
        )
        self.refresh()

    def on_predict_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        super().on_predict_end(trainer, pl_module)
        self._caching = False

    @property
    def predict_description(self) -> str:
        return "Caching" if self._caching else super().predict_description
