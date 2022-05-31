import warnings

from typing import Optional, Union, Sized, Iterable, Dict

import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from torch.utils.data import Dataset
from pytorch_lightning.callbacks import ModelSummary, EarlyStopping
from quaterion_models import SimilarityModel

from quaterion.dataset.similarity_data_loader import (
    PairsSimilarityDataLoader,
    GroupSimilarityDataLoader,
    SimilarityDataLoader,
)
from quaterion.eval.evaluator import Evaluator
from quaterion.loss import GroupLoss, PairwiseLoss
from quaterion.train.trainable_model import TrainableModel
from quaterion.train.callbacks import CleanupCallback, MetricsCallback
from quaterion.utils.enums import TrainStage
from quaterion.utils.progress_bar import QuaterionProgressBar


class Quaterion:
    """A dwarf on a giant's shoulders sees farther of the two"""

    @classmethod
    def fit(
        cls,
        trainable_model: TrainableModel,
        train_dataloader: SimilarityDataLoader,
        val_dataloader: Optional[SimilarityDataLoader] = None,
        trainer: Optional[pl.Trainer] = None,
        ckpt_path: Optional[str] = None,
    ):
        """Handle training routine

        Assemble data loaders, performs caching and whole training process.

        Args:
            trainable_model: model to fit
            train_dataloader: DataLoader instance to retrieve samples during training
                stage
            val_dataloader: Optional DataLoader instance to retrieve samples during
                validation stage
            trainer: `pytorch_lightning.Trainer` instance to handle fitting routine
                internally. If not passed will be created with `Quaterion.trainer_defaults`
            ckpt_path: Path/URL of the checkpoint from which training is resumed. If there is
                no checkpoint file at the path, an exception is raised. If resuming from mid-epoch checkpoint,
                training will start from the beginning of the next epoch.
        """

        if isinstance(train_dataloader, PairsSimilarityDataLoader):
            if not isinstance(trainable_model.loss, PairwiseLoss):
                raise NotImplementedError(
                    "Can't use PairsSimilarityDataLoader with non-PairwiseLoss"
                )

        if isinstance(train_dataloader, GroupSimilarityDataLoader):
            if not isinstance(trainable_model.loss, GroupLoss):
                raise NotImplementedError(
                    "Pair samplers are not implemented yet. "
                    "Try other loss/data loader"
                )

        if trainer is None:
            trainer = pl.Trainer(
                **cls.trainer_defaults(trainable_model=trainable_model, train_dataloader=train_dataloader)
            )

        trainer.callbacks.append(CleanupCallback())
        trainer.callbacks.append(MetricsCallback())
        # Prepare data loaders for training

        trainable_model.setup_dataloader(train_dataloader)
        if val_dataloader:
            trainable_model.setup_dataloader(val_dataloader)

        trainable_model.setup_cache(
            trainer=trainer,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
        )

        with warnings.catch_warnings():
            if train_dataloader.full_cache_used:
                warnings.filterwarnings(
                    "ignore", category=PossibleUserWarning, message="The dataloader.*"
                )

            trainer.fit(
                model=trainable_model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
                ckpt_path=ckpt_path,
            )

    @classmethod
    def evaluate(
        cls,
        evaluator: Evaluator,
        dataset: Union[Sized, Iterable, Dataset],
        model: SimilarityModel,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute metrics on a dataset

        Args:
            evaluator: Object which holds the configuration of which metrics to use and how to obtain samples for them
            dataset: Sized object, like list, tuple, torch.utils.data.Dataset, etc. to compute metrics
            model: SimilarityModel instance to perform objects encoding

        Returns:
            Dict[str, torch.Tensor] - dict of computed metrics.
            Where key - name of the metric and value - metric estimated values

        """
        return evaluator.evaluate(dataset, model)

    @staticmethod
    def trainer_defaults(trainable_model: TrainableModel = None, train_dataloader: SimilarityDataLoader = None):
        """Reasonable default parameters for `pytorch_lightning.Trainer`

        This function generates parameter set for Trainer, which are considered
        "recommended" for most use-cases of Quaterion.
        Quaterion similarity learning train process has characteristics that differentiate it from
        regular deep learning model training.
        This default parameters may be overwritten, if you need some special behaviour for your special task.

        Args:
            trainable_model: We will try to adjust default params based on model configuration, if provided
            train_dataloader: If provided, trainer params will be adjusted according to dataset

        Returns:
            kwargs for `pytorch_lightning.Trainer`
        """
        use_gpu = torch.cuda.is_available()
        defaults = {
            "callbacks": [
                QuaterionProgressBar(console_kwargs={"tab_size": 4}),
                EarlyStopping(f"{TrainStage.VALIDATION}_loss"),
                ModelSummary(max_depth=3),
            ],
            "gpus": int(use_gpu),
            "auto_select_gpus": use_gpu,
            "max_epochs": -1,
            "enable_model_summary": False  # We define our custom model summary
        }

        # Adjust default parameters according to the dataloader configuration
        if train_dataloader:
            try:
                num_batches = len(train_dataloader)
                if num_batches > 0:
                    defaults["log_every_n_steps"] = min(50, num_batches)
            except Exception:  # If dataset has to length
                pass

        # Adjust default parameters according to model configuration
        if trainable_model:
            # If the cache is enabled and there are no
            # trainable encoders - checkpointing on each epoch might become a bottleneck
            disable_checkpoints = all(
                [
                    not encoder.trainable
                    for encoder in trainable_model.model.encoders.values()
                ]
            ) and (trainable_model.configure_caches() is not None)

            if disable_checkpoints:
                defaults["enable_checkpointing"] = False
        return defaults
