from typing import (
    Union,
    Dict,
    Optional,
)

import pytorch_lightning as pl
import torch.cuda
from loguru import logger
from quaterion_models.encoders import Encoder
from torch.utils.data import DataLoader

from quaterion.train.cache.cache_train_collater import CacheTrainCollater
from quaterion.dataset.similarity_data_loader import SimilarityDataLoader
from quaterion.train.cache import (
    CacheConfig,
    CacheEncoder,
    CacheType,
    InMemoryCacheEncoder,
)
from quaterion.train.cache.cache_encoder import CacheMode
from quaterion.train.cache.cache_model import CacheModel


class CacheMixin:
    @classmethod
    def _apply_cache_config(
        cls,
        encoders: Union[Encoder, Dict[str, Encoder]],
        cache_config: Optional[CacheConfig],
    ) -> Union[Encoder, Dict[str, Encoder]]:
        """Applies received cache configuration for cached encoders, remain
        non-cached encoders as is

        Args:
            encoders: all model's encoders
            cache_config: CacheConfig instance defined in `configure_cache`
                method of the model

        Returns:
            Union[Encoder, Dict[str, encoder]]: encoder or dict of encoders
                which were wrapped into CacheEncoder instances according to
                received cache config.
                Result type depends on the way encoder was defined in the
                model: with or without explicit mapping

        Raises:
            KeyError: encoder's name in cache config is not in model's
                encoders
            ValueError: if CacheConfig instance does not have some of required
                options set. E.g. not `mapping` nor `cache_type` being set
        """
        if not cache_config:
            return encoders

        if not cache_config.cache_type and not cache_config.mapping:
            raise ValueError(
                "If cache is configured, cache_type or mapping have to be set"
            )

        if isinstance(encoders, Encoder):
            return cls._wrap_encoder(
                encoders,
                cache_config=cache_config,
            )

        cached_encoders = {}
        for encoder_name, encoder in encoders.items():
            cached_encoders[encoder_name] = cls._wrap_encoder(
                encoder, cache_config=cache_config, encoder_name=encoder_name
            )

        return {**encoders, **cached_encoders}

    @staticmethod
    def _check_cuda(cache_type: CacheType, encoder_name: str) -> None:
        if cache_type == CacheType.GPU and not torch.cuda.is_available():
            raise ValueError(
                f"`CacheType.GPU` has been chosen for `{encoder_name}` "
                "encoder, but cuda is not available"
            )

    @classmethod
    def _wrap_encoder(
        cls, encoder: Encoder, cache_config: CacheConfig, encoder_name: str = ""
    ) -> Encoder:
        """Wrap encoder into CacheEncoder instance if it is required by config.

        Args:
            encoder: raw model's encoder
            cache_config: cache type of tensor storage

        Returns:
            wrapped CacheEncoder or original encoder

        Raises:
            ValueError: if encoder layers are not frozen. Cache can be applied
                only to fully frozen encoders' outputs.
        """
        if encoder.trainable():
            if encoder_name in cache_config.mapping:
                raise ValueError(
                    f"Can't configure cache for encoder {encoder_name}. "
                    "Encoder must be frozen to cache it"
                )
            return encoder

        cache_type = cache_config.mapping.get(encoder_name) or cache_config.cache_type

        if cache_type is None:
            logger.info(
                f"{encoder_name} haven't been cached, "
                "but could be as non-trainable encoders"
            )
            return encoder

        cls._check_cuda(cache_type, encoder_name)

        return InMemoryCacheEncoder(encoder, cache_type)

    @classmethod
    def _cache(
        cls,
        trainer: pl.Trainer,
        encoders: Dict[str, Encoder],
        train_dataloader: SimilarityDataLoader,
        val_dataloader: Optional[SimilarityDataLoader],
        cache_config: CacheConfig,
    ) -> None:
        """Filling cache for model's cache encoders.

        Args:
            trainer: Lightning Trainer holds required parameters for model launch (gpu, e.t.c.)
            encoders: mapping of model's encoders and their names
            train_dataloader: model's train dataloader
            val_dataloader: model's val dataloader
            cache_config: cache config instance to configure cache batch size
                and num of workers to use for caching

        """
        cache_encoders = {
            name: encoder
            for name, encoder in encoders.items()
            if isinstance(encoder, CacheEncoder)
        }

        if not cache_encoders:
            return

        if cache_config.key_extractors and not isinstance(
            cache_config.key_extractors, dict
        ):
            # If only one function specified, use it for all encoders
            key_extractors = {
                name: cache_config.key_extractors for name in cache_encoders.keys()
            }
        else:
            key_extractors = cache_config.key_extractors

        cache_collater = CacheTrainCollater(
            pre_collate_fn=train_dataloader.pre_collate_fn,
            encoder_collates={
                name: encoder.get_collate_fn() for name, encoder in encoders.items()
            },
            key_extractors=key_extractors,
            cachable_encoders=list(cache_encoders.keys()),
            mode=CacheMode.FILL,
        )

        train_dataloader.collate_fn = cache_collater

        cache_train_dataloader = cls._wrap_cache_dataloader(
            dataloader=train_dataloader, cache_config=cache_config
        )

        cache_val_dataloader = None
        if val_dataloader is not None:
            val_dataloader.collate_fn = cache_collater
            cache_val_dataloader = cls._wrap_cache_dataloader(
                dataloader=val_dataloader, cache_config=cache_config
            )

        cls._fill_cache(
            trainer, cache_encoders, cache_train_dataloader, cache_val_dataloader
        )

        cache_collater.mode = CacheMode.TRAIN
        logger.info("Caching has been successfully finished")

    @classmethod
    def _fill_cache(
        cls,
        trainer: pl.Trainer,
        cache_encoders: Dict[str, CacheEncoder],
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
    ) -> None:
        """Fills cache and restores trainer state for further training process.

        Args:
            trainer: performs one training and validation epoch
            cache_encoders: mapping of encoders to cache input
            train_dataloader: model's train dataloader
            val_dataloader: model's val dataloader

        """
        # The actual caching
        trainer.predict(
            CacheModel(
                cache_encoders,
            ),
            [train_dataloader, val_dataloader],
        )

    @classmethod
    def _unwrap_cache_encoders(cls, encoders: Dict[str, Encoder]) -> Dict[str, Encoder]:
        unwrapped_encoders = {}
        for key, encoder in encoders.items():
            if isinstance(encoder, CacheEncoder):
                unwrapped_encoders[key] = encoder.wrapped_encoder
            else:
                unwrapped_encoders[key] = encoder
        return unwrapped_encoders

    @classmethod
    def _wrap_cache_dataloader(
        cls,
        dataloader: SimilarityDataLoader,
        cache_config: CacheConfig,
    ) -> DataLoader:
        """Creates dataloader for caching.

        Args:
            dataloader: dataloader to be wrapped
            cache_config: cache config to retrieve num of workers and batch
                size

        Returns:
            DataLoader: dataloader for caching
        """
        num_workers = (
            cache_config.num_workers
            if cache_config.num_workers is not None
            else dataloader.num_workers
        )

        # We need to reduce random sampling and repeated calculations to
        # make cache as fast as possible. Thus, we recreate dataloader
        # and set batch size explicitly.
        params = {
            **dataloader.original_params,
            "num_workers": num_workers,
            "batch_size": cache_config.batch_size,
            "shuffle": False,
            "sampler": None,
        }

        params.pop("collate_fn")  # Explicitly override collate

        cache_dl = DataLoader(
            dataset=dataloader.dataset, collate_fn=dataloader.collate_fn, **params
        )
        return cache_dl
