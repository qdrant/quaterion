from typing import Union, Dict, Optional, Set, Any, Callable, Hashable

import torch.cuda
from loguru import logger
from torch.utils.data import DataLoader
from quaterion_models.encoders import Encoder
from quaterion_models.model import DEFAULT_ENCODER_KEY

from quaterion.train.encoders import (
    CacheConfig,
    CacheEncoder,
    CacheType,
    InMemoryCacheEncoder,
)


class CacheMixin:
    @classmethod
    def apply_cache_config(
        cls,
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
                if not encoders[encoder_name].trainable()
            }

            for encoder_name, cache_type in cache_config.mapping.items():
                encoder: Optional[Encoder] = encoders.get(encoder_name)
                if not encoder:
                    raise KeyError(
                        f"Can't configure cache for encoder {encoder_name}. "
                        "Encoder not found"
                    )
                cls._check_cuda(cache_type, encoder_name)
                key_extractor: Optional[
                    Callable[[Any], Hashable]
                ] = cache_config.key_extractors.get(encoder_name)

                encoders[encoder_name]: CacheEncoder = cls.wrap_encoder(
                    encoder, cache_type, key_extractor, encoder_name,
                )

                possible_cache_encoders.remove(encoder_name)

            not_cached_encoders = ", ".join(possible_cache_encoders)
            if not_cached_encoders:
                logger.info(
                    f"{not_cached_encoders} haven't been cached, "
                    "but could be as non-trainable encoders"
                )

        elif cache_config.cache_type:
            encoder_name = DEFAULT_ENCODER_KEY

            cls._check_cuda(cache_config.cache_type, encoder_name)
            key_extractor = cache_config.key_extractors.get(encoder_name)
            encoders = cls.wrap_encoder(
                encoders, cache_config.cache_type, key_extractor, encoder_name,
            )
        else:
            raise ValueError(
                "If cache is configured, cache_type or mapping have to be set"
            )

        return encoders

    @staticmethod
    def _check_cuda(cache_type, encoder_name):
        if cache_type == CacheType.GPU and not torch.cuda.is_available():
            raise ValueError(
                f"`CacheType.GPU` has been chosen for `{encoder_name}` "
                "encoder, but cuda is not available"
            )

    @staticmethod
    def wrap_encoder(
        encoder: Encoder,
        cache_type: CacheType,
        key_extractor: Optional[Callable[[Any], Hashable]],
        encoder_name: str = "",
    ) -> Encoder:
        if encoder.trainable():
            raise ValueError(
                f"Can't configure cache for encoder {encoder_name}. "
                "Encoder must be frozen to cache it"
            )

        encoder = InMemoryCacheEncoder(encoder, cache_type)

        if key_extractor:
            encoder.configure_key_extractor(key_extractor)

        return encoder

    @staticmethod
    def cache(
        encoders, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader],
    ) -> None:
        """
        Fill cache for each CacheEncoder

        :param encoders: MetricModel
        :param train_dataloader:
        :param val_dataloader:
        :return: None
        """
        cache_encoders = {
            name: encoder
            for name, encoder in encoders.items()
            if isinstance(encoder, CacheEncoder)
        }

        if not cache_encoders:
            return

        def cache_dataloader(dataloader):
            for sample in dataloader:
                features, _ = sample
                for name, encoder in cache_encoders.items():
                    encoder.fill_cache(features[name])

        cache_dataloader(train_dataloader)
        val_dataloader = val_dataloader if val_dataloader is not None else []
        cache_dataloader(val_dataloader)

        # Once cache is filled, collate functions return only keys for cache
        for encoder_name in cache_encoders:
            encoders[encoder_name].cache_filled = True
