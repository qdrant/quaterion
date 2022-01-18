from typing import Dict, Any, Union, Optional, Set
from loguru import logger

import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from quaterion_models.encoder import Encoder
from quaterion_models.heads.encoder_head import EncoderHead
from quaterion_models.model import MetricModel

from quaterion.train.encoders import (
    CacheConfig,
    CacheType,
    CpuCacheEncoder,
    GpuCacheEncoder,
)
from quaterion.train.encoders.cache_encoder import CacheEncoder
from quaterion.loss.similarity_loss import SimilarityLoss
from quaterion.utils.enums import TrainStage


class TrainableModel(pl.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        encoders = self.configure_encoders()
        self.cache_config = self.configure_caches()
        encoders = self._apply_cache_config(encoders, self.cache_config)

        head = self.configure_head(
            MetricModel.get_encoders_output_size(encoders)
        )

        self._model = MetricModel(encoders=encoders, head=head)
        self._loss = self.configure_loss()

    def _apply_cache_config(
        self,
        encoders: Union[Encoder, Dict[str, Encoder]],
        cache_config: Optional[CacheConfig],
    ) -> Union[Encoder, Dict[str, Encoder]]:
        """
        Applies received cache configuration for cached encoders, remain
        non-cached encoders as is
        """
        if not cache_config:
            return encoders

        if cache_config.mapping:
            if cache_config.cache_type:
                logger.warning(
                    "CacheConfig.cache_type has no effect when mapping is set"
                )

            possible_cache_encoders: Set[str] = {
                encoder_name
                for encoder_name in encoders
                if encoders[encoder_name].trainable()
            }

            for encoder_name, cache_type in cache_config.mapping.items():
                encoder: Optional[Encoder] = encoders.get(encoder_name)
                if not encoder:
                    raise KeyError(
                        f"Can't configure cache for encoder {encoder_name}. "
                        "Encoder not found"
                    )
                encoders[encoder_name]: CacheEncoder = self._wrap_encoder(
                    encoder, cache_type, encoder_name
                )
                possible_cache_encoders.remove(encoder_name)

            not_cached_encoders = ", ".join(possible_cache_encoders)
            if not_cached_encoders:
                logger.info(
                    f"{not_cached_encoders} haven't been cached, "
                    "but could be as non-trainable encoders"
                )

        elif cache_config.cache_type:
            encoders = self._wrap_encoder(encoders, cache_config.cache_type)
        else:
            raise ValueError(
                "If cache is configured, cache_type or mapping have to be set"
            )

        return encoders

    def _wrap_encoder(
        self, encoder: Encoder, cache_type: CacheType, encoder_name: str = ""
    ) -> Encoder:
        if encoder.trainable():
            raise ValueError(
                f"Can't configure cache for encoder {encoder_name}. "
                "Encoder must be frozen to cache it"
            )

        if cache_type == CacheType.AUTO:
            cache_wrapper = (
                GpuCacheEncoder
                if torch.cuda.is_available()
                else CpuCacheEncoder
            )
            encoder = cache_wrapper(encoder)
        elif cache_type == CacheType.CPU:
            encoder = CpuCacheEncoder(encoder)
        elif cache_type == CacheType.GPU:
            encoder = GpuCacheEncoder(encoder)

        key_extractor = self.cache_config.key_extractors.get(
            encoder_name
        )
        if key_extractor:
            encoder.configure_(key_extractor)

        return encoder

    @property
    def model(self) -> MetricModel:
        return self._model

    @property
    def loss(self) -> SimilarityLoss:
        return self._loss

    def configure_loss(self) -> SimilarityLoss:
        raise NotImplementedError()

    def configure_encoders(self) -> Union[Encoder, Dict[str, Encoder]]:
        """
        Use this function to define an initial state of encoders.
        This function should be used to assign initial values for encoders
        before training as well as during the checkpoint loading.

        :return: Instance of the `Encoder` or dict of instances
        """
        raise NotImplementedError()

    def configure_caches(self) -> Optional[CacheConfig]:
        """
        Use this method to define which encoders should cache calculated
        embeddings and what kind of cache they should use.

        Examples:

        >>> CacheConfig(CacheType.AUTO)
        CacheConfig(cache_type=<CacheType.AUTO: 'auto'>, mapping={}, key_extractors={})

        >>> cache_config = CacheConfig(
...     mapping={"text_encoder": CacheType.GPU, "image_encoder": CacheType.CPU}
... )
        CacheConfig(
            cache_type=None,
            mapping={
                'text_encoder': <CacheType.GPU: 'gpu'>,
                'image_encoder': <CacheType.CPU: 'cpu'>
            },
            key_extractors={}
        )
        >>> CacheConfig(
...     cache_type=CacheType.AUTO,
...     key_extractors={"default": lambda obj: hash(obj)}
... )
        CacheConfig(
            cache_type=CacheType.AUTO,
            key_extractors={"default": lambda obj: hash(obj)}
        )

        :return: CacheConfig
        """
        pass

    def configure_head(self, input_embedding_size: int) -> EncoderHead:
        """
        Use this function to define an initial state for head layer of the model

        :param input_embedding_size: size of embeddings produced by encoders
        :return: Instance of `EncoderHead`
        """
        raise NotImplementedError()

    def process_results(
        self,
        embeddings: torch.Tensor,
        targets: Dict[str, Any],
        batch_idx,
        stage: TrainStage,
        **kwargs,
    ):
        """
        Define any additional evaluations of embeddings here.

        :param embeddings: Tensor of batch embeddings, shape: [batch_size x embedding_size]
        :param targets: Output of batch target collate
        :param batch_idx: ID of the processing batch
        :param stage: Train, validation or test stage
        :return: None
        """
        pass

    def cache(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader],
    ) -> None:
        """
        Fill cache for each CacheEncoder

        :param train_dataloader:
        :param val_dataloader:
        :return: None
        """
        cache_encoders = {
            name: encoder
            for name, encoder in self.model.encoders.items()
            if isinstance(encoder, CacheEncoder)
        }

        if not cache_encoders:
            return

        def cache_dataloader(dataloader, encoders):
            for sample in dataloader:
                features, _ = sample
                for name, encoder in encoders.items():
                    encoder.fill_cache(features[name])

        cache_dataloader(train_dataloader, cache_encoders)
        val_dataloader = val_dataloader if val_dataloader is not None else []
        cache_dataloader(val_dataloader, cache_encoders)

        # Once cache is filled, collate functions return only keys for cache
        for encoder_name in cache_encoders:
            self.model.encoders[encoder_name].cache_filled = True

    def training_step(self, batch, batch_idx, **kwargs) -> torch.Tensor:
        stage = TrainStage.TRAIN
        loss = self._common_step(
            batch=batch, batch_idx=batch_idx, stage=stage, **kwargs
        )
        return loss

    def validation_step(
        self, batch, batch_idx, **kwargs
    ) -> Optional[torch.Tensor]:
        stage = TrainStage.VALIDATION
        self._common_step(
            batch=batch, batch_idx=batch_idx, stage=stage, **kwargs
        )
        return None

    def test_step(self, batch, batch_idx, **kwargs) -> Optional[torch.Tensor]:
        stage = TrainStage.TEST
        self._common_step(
            batch=batch, batch_idx=batch_idx, stage=stage, **kwargs
        )
        return None

    def _common_step(self, batch, batch_idx, stage: TrainStage, **kwargs):
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
        """
        Save model, independent of Pytorch Lightning.

        :param path: where to save
        :return: None
        """
        self.model.save(path)
