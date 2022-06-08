from rich.console import RenderableType
from rich.table import Column
from rich.text import Text
from tkinter.ttk import Style

from rich.progress import TextColumn
from typing import Optional, Dict, Any, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress.rich_progress import (
    RichProgressBar,
    RichProgressBarTheme,
    CustomBarColumn,
    BatchesProcessedColumn,
    CustomTimeColumn,
    ProcessingSpeedColumn,
)

from quaterion.train.cache.cache_model import CacheModel


class FixedLengthProcessionSpeed(ProcessingSpeedColumn):
    """Renders processing speed for the progress bar with fixes length"""

    def __init__(self, style: Union[str, Style]):
        super().__init__(style)
        self.max_length = len("0.00")

    def render(self, task) -> RenderableType:
        task_speed = f"{task.speed:>.2f}" if task.speed is not None else "0.00"
        self.max_length = max(len(task_speed), self.max_length)
        task_speed = " " * (self.max_length - len(task_speed)) + task_speed
        return Text(f"{task_speed}it/s", style=self.style, justify="center")


class QuaterionProgressBar(RichProgressBar):
    def __init__(
        self,
        refresh_rate: int = 1,
        leave: bool = False,
        theme: RichProgressBarTheme = None,
        console_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if theme is None:
            theme = RichProgressBarTheme(
                description="white",
                progress_bar="#4881AD",
                progress_bar_finished="#67C87A",
                progress_bar_pulse="#67C87A",
                batch_progress="white",
                time="grey54",
                processing_speed="grey70",
                metrics="white",
            )

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

    def configure_columns(self, trainer) -> list:
        return [
            TextColumn(
                "[progress.description]{task.description}",
                table_column=Column(
                    no_wrap=True,
                    min_width=9,  # prevents blinking during validation, length of `Validation `
                ),
            ),
            CustomBarColumn(
                complete_style=self.theme.progress_bar,
                finished_style=self.theme.progress_bar_finished,
                pulse_style=self.theme.progress_bar_pulse,
            ),
            BatchesProcessedColumn(style=self.theme.batch_progress),
            CustomTimeColumn(style=self.theme.time),
            FixedLengthProcessionSpeed(style=self.theme.processing_speed),
        ]
