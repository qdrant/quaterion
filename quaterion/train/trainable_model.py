from __future__ import annotations
from typing import Dict, Any, Union, Optional, List

import pytorch_lightning as pl

from quaterion_models import SimilarityModel
from quaterion_models.encoders import Encoder
from quaterion_models.heads import EncoderHead
from quaterion_models.types import TensorInterchange
from torch import Tensor

from quaterion.dataset import SimilarityDataLoader
from quaterion.dataset.train_collator import TrainCollator
from quaterion.eval.attached_metric import AttachedMetric
from quaterion.loss import SimilarityLoss
from quaterion.train.cache import (
    CacheConfig,
    CacheType,
)
from quaterion.train.cache_mixin import CacheMixin
from quaterion.utils.enums import TrainStage


class TrainableModel(pl.LightningModule, CacheMixin):
    """Base class for models to be trained.

    TrainableModel is used to describe how and which components of the model should be trained.

    It assembles model from building blocks like
    :class:`~quaterion_models.encoders.encoder.Encoder`,
    :class:`~quaterion_models.heads.encoder_head.EncoderHead`, etc.

    .. code-block:: none

         ┌─────────┐ ┌─────────┐ ┌─────────┐
         │Encoder 1│ │Encoder 2│ │Encoder 3│
         └────┬────┘ └────┬────┘ └────┬────┘
              │           │           │
              └────────┐  │  ┌────────┘
                       │  │  │
                   ┌───┴──┴──┴───┐
                   │   concat    │
                   └──────┬──────┘
                          │
                   ┌──────┴──────┐
                   │    Head     │
                   └─────────────┘

    TrainableModel also handles the majority of the training process routine: training and
    validation steps, tensors device management, logging, and many more.
    Most of the training routines are inherited from
    :class:`~pytorch_lightning.LightningModule`, which is a direct ancestor of TrainableModel.

    To train a model you need to inherit it from TrainableModel and implement required methods and
    attributes.

    Minimal Example::

        class ExampleModel(TrainableModel):
            def __init__(self, lr=10e-5, *args, **kwargs):
                self.lr = lr
                super().__init__(*args, **kwargs)

            # backbone of the model
            def configure_encoders(self):
                return YourAwesomeEncoder()

            # top layer of the model
            def configure_head(self, input_embedding_size: int):
                return SkipConnectionHead(input_embedding_size)

            def configure_optimizers(self):
                return Adam(self.model.parameters(), lr=self.lr)

            def configure_loss(self):
                return ContrastiveLoss()


    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        encoders = self.configure_encoders()
        self.cache_config: CacheConfig = self.configure_caches() or CacheConfig(
            cache_type=CacheType.NONE
        )

        head = self.configure_head(SimilarityModel.get_encoders_output_size(encoders))

        metrics = self.configure_metrics()
        self.attached_metrics: List[AttachedMetric] = (
            [metrics] if isinstance(metrics, AttachedMetric) else metrics
        )

        self._model = SimilarityModel(encoders=encoders, head=head)
        self._loss = self.configure_loss()

    def configure_metrics(self) -> Union[AttachedMetric, List[AttachedMetric]]:
        """Method to configure batch-wise metrics for a training process

        Use this method to attach batch-wise metrics to a training process.
        Provided metrics have to have similar to
        :class:`~quaterion.eval.pair.pair_metric.PairMetric` or
        :class:`~quaterion.eval.group.group_metric.GroupMetric`

        Returns:
            Union[:class:`~quaterion.eval.attached_metric.AttachedMetric`,
            List[:class:`~quaterion.eval.attached_metric.AttachedMetric`]] -
            metrics attached to the model

        Examples::

            return [
                AttachedMetric(
                    "RetrievalPrecision",
                    RetrievalPrecision(k=1),
                    prog_bar=True,
                    on_epoch=True,
                ),
                AttachedMetric(
                    "RetrievalReciprocalRank",
                    RetrievalReciprocalRank(),
                    prog_bar=True,
                ),
            ]
        """
        return []

    def _evaluate(
        self,
        embeddings: Tensor,
        targets: Dict[str, Any],
        stage: TrainStage,
    ) -> None:
        """Method to calculate and log metrics, accumulate embeddings in estimators

        Calculate current stage and batch metrics

        Args:
            embeddings: current batch embeddings
            targets: objects to calculate labels for similarity samples
            stage: training, validation, etc.
        """
        for metric in self.attached_metrics:
            if stage in metric.stages:
                self.log(
                    f"{metric.name}_{stage}",
                    metric.compute(embeddings, **targets),
                    **metric.log_options,
                    batch_size=embeddings.shape[0],
                )

    @property
    def model(self) -> SimilarityModel:
        """Origin model to be trained

        Returns:
            :class:`~quaterion_models.model.SimilarityModel`: model to be trained
        """
        return self._model

    @property
    def loss(self) -> SimilarityLoss:
        """Property to get the loss function to use."""
        return self._loss

    def configure_loss(self) -> SimilarityLoss:
        """Method to configure loss function to use."""
        raise NotImplementedError()

    def configure_encoders(self) -> Union[Encoder, Dict[str, Encoder]]:
        """Method to provide encoders configuration

        Use this function to define an initial state of encoders.
        This function should be used to assign initial values for encoders
        before training as well as during the checkpoint loading.

        Returns:
            Union[:class:`~quaterion_models.encoders.encoder.Encoder`,
            Dict[str, :class:`~quaterion_models.encoders.encoder.Encoder`]]:
            instance of encoder which will be assigned to
            :const:`~quaterion_models.model.DEFAULT_ENCODER_KEY`, or mapping of names and
            encoders.
        """
        raise NotImplementedError()

    def configure_caches(self) -> Optional[CacheConfig]:
        """Method to provide cache configuration

        Use this method to define which encoders should cache calculated embeddings and
        what kind of cache they should use.

        Returns:
            Optional[:class:`~quaterion.train.cache.cache_config.CacheConfig`]: cache configuration
            to be applied if provided, None otherwise.

        Examples:

        `Do not use cache (default)`::

            return None

        `Configure cache automatically for all non-trainable encoders`::

            return CacheConfig(CacheType.AUTO)

        `Specify cache type for each encoder individually`::

            return CacheConfig(mapping={
                    "text_encoder": CacheType.GPU,
                    # Store cache in GPU for `text_encoder`
                    "image_encoder": CacheType.CPU
                    # Store cache in RAM for `image_encoder`
                }
            )

        `Specify key for cache object disambiguation`::

            return CacheConfig(
                cache_type=CacheType.AUTO,
                key_extractors={"text_encoder": hash}
            )


        This function might be useful if you want to provide some more sophisticated way of
        storing association between cached vectors and original object.
        Item numbers from dataset
        will be used by default if key is not specified.

        """
        pass

    def configure_head(self, input_embedding_size: int) -> EncoderHead:
        """Use this function to define an initial state for head layer of the model.

        Args:
            input_embedding_size: size of embeddings produced by encoders
        Returns:
            :class:`~quaterion_models.heads.encoder_head.EncoderHead`: head to be added on top of
            a model
        """
        raise NotImplementedError()

    def process_results(
        self,
        embeddings: Tensor,
        targets: Dict[str, Any],
        batch_idx: int,
        stage: TrainStage,
        **kwargs,
    ):
        """Method to provide any additional evaluations of embeddings.

        Args:
            embeddings: shape: (batch_size, embedding_size) - model's output.
            targets: output of batch target collate.
            batch_idx: ID of the processing batch.
            stage: train, validation or test stage.
        """
        pass

    def training_step(
        self, batch: TensorInterchange, batch_idx: int, **kwargs
    ) -> Tensor:
        """:meta private: Compute and return the training loss and some additional metrics for e.g.
        the progress bar or logger.

        Args:
            batch: Output of DataLoader.
            batch_idx: Integer displaying index of this batch.
            **kwargs: keyword arguments to be passed into :meth:`~process_results`

        Returns:
            Tensor: computed loss value
        """
        stage = TrainStage.TRAIN
        loss = self._common_step(
            batch=batch, batch_idx=batch_idx, stage=stage, **kwargs
        )
        return loss

    def validation_step(self, batch, batch_idx, **kwargs) -> Optional[Tensor]:
        """:meta private: Compute validation loss and some additional metrics for e.g. the progress
        bar or logger.

        Args:
            batch: Output of DataLoader.
            batch_idx: Integer displaying index of this batch.
            **kwargs: keyword arguments to be passed into :meth:`~process_results`
        """
        stage = TrainStage.VALIDATION
        self._common_step(batch=batch, batch_idx=batch_idx, stage=stage, **kwargs)
        return None

    def test_step(self, batch, batch_idx, **kwargs) -> Optional[Tensor]:
        """:meta private: Compute test loss and some additional metrics for e.g. the progress
        bar or logger.

        Args:
            batch: Output of DataLoader.
            batch_idx: Integer displaying index of this batch.
            **kwargs: keyword arguments to be passed into :meth:`~process_results`
        """
        stage = TrainStage.TEST
        self._common_step(batch=batch, batch_idx=batch_idx, stage=stage, **kwargs)
        return None

    def _common_step(self, batch, batch_idx, stage: TrainStage, **kwargs) -> Tensor:
        """Common step to compute loss and metrics for training, validation, test
         and other stages.

        Args:
            batch: Output of DataLoader.
            batch_idx: Integer displaying index of this batch.
            stage: current training stage: training, validation, etc.
            **kwargs: keyword arguments to be passed into :meth:`~process_results`

        Returns:
            Tensor: computed loss value
        """
        features, targets = batch

        embeddings = self.model(features)
        loss = self.loss(embeddings=embeddings, **targets)
        self.log(f"{stage}_loss", loss, batch_size=embeddings.shape[0])

        self._evaluate(
            embeddings=embeddings,
            targets=targets,
            stage=stage,
        )

        self.process_results(
            embeddings=embeddings,
            targets=targets,
            batch_idx=batch_idx,
            stage=stage,
            **kwargs,
        )

        return loss

    def save_servable(self, path: str):
        """Save model for serving, independent of Pytorch Lightning

        Args:
            path: path to save to
        """
        self.unwrap_cache()
        self.model.save(path)

    def setup_cache(
        self,
        trainer: pl.Trainer,
        train_dataloader: SimilarityDataLoader,
        val_dataloader: Optional[SimilarityDataLoader],
    ):
        """:meta private: Prepares encoder's cache for faster training:

        - Replaces frozen encoders with cache Wrapper according
          to the cache configuration defined in :meth:`configure_caches`.
        - Fill cachable encoders with embeddings

        Args:
            trainer: Pytorch Lightning trainer object
            train_dataloader: train dataloader
            val_dataloader: validation dataloader

        Returns:
            `False`, if cache is already created or not required.
            `True`, if cache is newly created
        """
        self.model.encoders = self._apply_cache_config(
            self.model.encoders, self.cache_config
        )

        return self._cache(
            trainer=trainer,
            encoders=self.model.encoders,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            cache_config=self.cache_config,
        )

    def unwrap_cache(self):
        """:meta private: Restore original encoders"""
        self.model.encoders = self._unwrap_cache_encoders(self.model.encoders)

    def setup_dataloader(self, dataloader: SimilarityDataLoader):
        """Setup data loader for encoder-specific settings, Setup encoder-specific collate function

        Each encoder have its own unique way to transform a list of records into NN-compatible format.
        These transformations are usually done during data pre-processing step.
        """
        encoder_collate_fns = dict(
            (key, encoder.get_collate_fn())
            for key, encoder in self.model.encoders.items()
        )

        collator = TrainCollator(
            pre_collate_fn=dataloader.collate_fn,
            encoder_collates=encoder_collate_fns,
        )

        dataloader.collate_fn = collator
