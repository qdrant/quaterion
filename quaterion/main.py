import warnings

import torch
from typing import Optional, Union, Sized, Iterable, Dict

import pytorch_lightning as pl
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from torch.utils.data import Dataset
from quaterion_models import SimilarityModel

from quaterion.dataset.similarity_data_loader import (
    PairsSimilarityDataLoader,
    GroupSimilarityDataLoader,
    SimilarityDataLoader,
)
from quaterion.eval.evaluator import Evaluator
from quaterion.loss import GroupLoss, PairwiseLoss
from quaterion.train.cleanup_callback import CleanupCallback
from quaterion.train.metrics_callback import MetricsCallback
from quaterion.train.trainable_model import TrainableModel


class Quaterion:
    """A dwarf on a giant's shoulders sees farther of the two

    One of the core entities in the framework.
    Contains methods to launch the actual training and evaluation processes.

    Examples:

        `Default trainer run`::

            import pytorch_lightning as pl

            from quaterion import Quaterion
            from quaterion.dataset import PairsSimilarityDataLoader
            from quaterion.eval.evaluator import Evaluator
            from quaterion.eval.pair import RetrievalPrecision
            from quaterion.eval.samplers.pair_sampler import PairSampler


            def train(model, train_dataset_path, val_dataset_path):
                # region data setup
                train_dataset = YourDataset(train_dataset_path)
                val_dataset = YourDataset(val_dataset_path)
                train_dataloader = PairsSimilarityDataLoader(train_dataset, batch_size=1024)
                val_dataloader = PairsSimilarityDataLoader(val_dataset, batch_size=1024)
                # endregion

                # region fit
                # To use trainer with the defaults provided by Quaterion trainer should be
                # explicitly set to `None`
                Quaterion.fit(model, trainer=None, train_dataloader, val_dataloader)
                # endregion

                # region evaluation
                metrics = {
                    "rp@1": RetrievalPrecision(k=1)
                }
                sampler = PairSampler()
                evaluator = Evaluator(metrics, sampler)
                results = Quaterion.evaluate(evaluator, val_dataset, model.model)
                print(f"results: {results}")
                # endregion

        `Custom trainer run`::

            # the same imports
            ...

            def train(model, train_dataset_path, val_dataset_path):
                # the same data setup region
                ...

                # region fit
                trainer = pl.Trainer(
                    max_epochs=params.get("max_epochs", 500),
                    auto_select_gpus=True,
                    log_every_n_steps=50,
                    gpus=1,
                )
                Quaterion.fit(model, trainer, train_dataloader, val_dataloader)
                # endregion

                # the same evaluation region
                ...

        `Custom trainer run with Quaterion defaults`::

             # the same imports
             ...


             def train(model, train_dataset_path, val_dataset_path):
                # the same data setup region
                ...

                # region fit
                quaterion_defaults = Quaterion.trainer_defaults()
                quaterion_defaults['logger'] = pl.loggers.WandbLogger(
                    name="example_model",
                    project="example_project",
                )
                quaterion_defaults['callbacks'].append(YourCustomCallback())
                trainer = pl.Trainer(**quaterion_defaults)
                Quaterion.fit(model, trainer, train_dataloader, val_dataloader)
                # endregion

                # the same evaluation region
                ...
    """

    @classmethod
    def fit(
        cls,
        trainable_model: TrainableModel,
        trainer: pl.Trainer,
        train_dataloader: SimilarityDataLoader,
        val_dataloader: Optional[SimilarityDataLoader] = None,
        ckpt_path: Optional[str] = None,
    ):
        """Handle training routine

        Assemble data loaders, performs caching and whole training process.

        Args:
            trainable_model: model to fit
            trainer: `pytorch_lightning.Trainer` instance to handle fitting routine
                internally
            train_dataloader: DataLoader instance to retrieve samples during training
                stage
            val_dataloader: Optional DataLoader instance to retrieve samples during
                validation stage
            ckpt_path: Path/URL of the checkpoint from which training is resumed.
                If there is no checkpoint file at the path, an exception is raised.
                If resuming from mid-epoch checkpoint, training will start from the beginning of
                the next epoch.
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
            evaluator: Object which holds the configuration of which metrics to use and how to
                obtain samples for them
            dataset: Sized object, like list, tuple, torch.utils.data.Dataset, etc. to compute
                metrics
            model: SimilarityModel instance to perform objects encoding

        Returns:
            Dict[str, torch.Tensor] - dict of computed metrics.
            Where key - name of the metric and value - metric estimated values

        """
        return evaluator.evaluate(dataset, model)
