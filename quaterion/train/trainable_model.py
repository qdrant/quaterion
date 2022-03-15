from __future__ import annotations
from typing import Dict, Any, Union, Optional

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import (
    TRAIN_DATALOADERS,
    EVAL_DATALOADERS,
)
from quaterion_models import MetricModel
from quaterion_models.encoders import Encoder
from quaterion_models.heads import EncoderHead
from quaterion_models.types import TensorInterchange
from torch import Tensor

from quaterion.dataset import SimilarityDataLoader
from quaterion.dataset.train_collater import TrainCollater
from quaterion.loss import SimilarityLoss
from quaterion.train.cache import (
    CacheConfig,
)
from quaterion.train.cache_mixin import CacheMixin
from quaterion.utils.enums import TrainStage


class TrainableModel(pl.LightningModule, CacheMixin):
    """Base class for models to be trained."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        encoders = self.configure_encoders()
        self.cache_config = self.configure_caches()
        encoders = self._apply_cache_config(encoders, self.cache_config)

        head = self.configure_head(MetricModel.get_encoders_output_size(encoders))

        self._model = MetricModel(encoders=encoders, head=head)
        self._loss = self.configure_loss()

    @property
    def model(self) -> MetricModel:
        """Origin model to be trained

        Returns:
            :class:`~quaterion_models.model.MetricModel`: model to be trained
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
        """Compute and return the training loss and some additional metrics for e.g.
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
        """Compute validation loss and some additional metrics for e.g. the progress
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
        """Compute test loss and some additional metrics for e.g. the progress
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
        self.log(f"{stage}_loss", loss)
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

    def cache(
        self,
        trainer: pl.Trainer,
        train_dataloader: SimilarityDataLoader,
        val_dataloader: Optional[SimilarityDataLoader],
    ):
        """Fill cachable encoders with embeddings"""
        self._cache(
            trainer=trainer,
            encoders=self.model.encoders,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            cache_config=self.cache_config,
        )

    def unwrap_cache(self):
        """Restore original encoders"""
        self.model.encoders = self._unwrap_cache_encoders(self.model.encoders)

    def setup_dataloader(self, dataloader: SimilarityDataLoader):
        """Update data loader's collate function with encoder-specific collate"""
        encoder_collate_fns = dict(
            (key, encoder.get_collate_fn())
            for key, encoder in self.model.encoders.items()
        )

        collater = TrainCollater(
            pre_collate_fn=dataloader.collate_fn,
            encoder_collates=encoder_collate_fns,
        )

        dataloader.collate_fn = collater

    # region anchors
    # https://github.com/PyTorchLightning/pytorch-lightning/issues/10667
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """:meta private:"""
        pass

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """:meta private:"""
        pass

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """:meta private:"""
        pass

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        """:meta private:"""
        pass

    # endregion
