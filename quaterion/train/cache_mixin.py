import os
import multiprocessing as mp

from typing import Union, Dict, Optional, Set, Any, Callable, Hashable

import torch.cuda

from loguru import logger
from torch.utils.data import DataLoader
from quaterion_models.encoders import Encoder
from quaterion_models.model import DEFAULT_ENCODER_KEY

from quaterion.dataset import (
    PairsSimilarityDataLoader,
    GroupSimilarityDataLoader,
)
from quaterion.train.encoders import (
    CacheConfig,
    CacheEncoder,
    CacheType,
    InMemoryCacheEncoder,
)
from quaterion.train.encoders.cache_encoder import CacheCollateFnType


class CacheMixin:
    # Child processes need to derive randomized `PYTHONHASHSEED` value from
    # parent process. It is only done with `fork` start method.
    # `fork` start method is presented on Unix systems and it is the
    # default for them, except macOS. Therefore, we set `fork` method
    # explicitly and cache is not supported on Windows.
    CACHE_MULTIPROCESSING_CONTEXT = "fork"

    @classmethod
    def _apply_cache_config(
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

                encoders[encoder_name]: CacheEncoder = cls._wrap_encoder(
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
            encoders = cls._wrap_encoder(
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
    def _wrap_encoder(
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

    @classmethod
    def cache(
        cls,
        encoders,
        train_dataloader: Union[
            PairsSimilarityDataLoader, GroupSimilarityDataLoader
        ],
        val_dataloader: Optional[
            Union[PairsSimilarityDataLoader, GroupSimilarityDataLoader]
        ],
        model_collate_fn: CacheCollateFnType,
        cache_config: CacheConfig,
    ) -> None:
        """
        Fill cache for each CacheEncoder

        :param encoders: MetricModel
        :param train_dataloader:
        :param val_dataloader:
        :param model_collate_fn:
        :param cache_config:
        :return: None
        """
        cache_encoders = {
            name: encoder
            for name, encoder in encoders.items()
            if isinstance(encoder, CacheEncoder)
        }

        if not cache_encoders:
            return

        cls._compatibility_check(train_dataloader)
        train_dataloader.set_model_collate_fn(model_collate_fn)
        cls._cache_dataloader(train_dataloader, cache_config, cache_encoders)

        if val_dataloader is not None:
            val_dataloader.set_model_collate_fn(model_collate_fn)
            cls._cache_dataloader(val_dataloader, cache_config, cache_encoders)

        # Once cache is filled, collate functions return only keys for cache
        for encoder_name in cache_encoders:
            encoders[encoder_name].cache_filled = True

    @classmethod
    def _compatibility_check(cls, dataloader: DataLoader):
        """
        Cache compatible only with dataloaders which have an implementation
        of `cache_collate_fn`
        """
        if not hasattr(dataloader, "cache_collate_fn"):
            raise TypeError(
                "DataLoader must have an implementation of `cache_collate_fn`"
                " to be cached."
            )

    @classmethod
    def _cache_dataloader(
        cls,
        dataloader: DataLoader,
        cache_config: CacheConfig,
        cache_encoders: Dict[str, CacheEncoder],
    ):
        """
        Fills cache for provided dataloader, switches multiprocessing context
        if required
        """
        cls._switch_multiprocessing_context(dataloader)
        num_workers = (
            cache_config.num_workers
            if cache_config.num_workers is not None
            else dataloader.num_workers
        )

        if num_workers == 0:
            mp_ctx = None
        elif dataloader.multiprocessing_context:  # already switched or
            # set by user
            mp_ctx = dataloader.multiprocessing_context
        elif "PYTHONHASHSEED" in os.environ:  # source dataloader has no
            # mp context set, use default on current OS
            mp_ctx = mp.get_start_method()
        else:
            mp_ctx = cls.CACHE_MULTIPROCESSING_CONTEXT
            cls._check_mp_context(mp_ctx)

        cache_collate_fn = getattr(dataloader, "cache_collate_fn")
        cache_dl = DataLoader(
            dataset=dataloader.dataset,
            batch_size=cache_config.batch_size,
            collate_fn=cache_collate_fn,
            num_workers=num_workers,
            pin_memory=dataloader.pin_memory,
            timeout=dataloader.timeout,
            worker_init_fn=dataloader.worker_init_fn,
            prefetch_factor=dataloader.prefetch_factor,
            multiprocessing_context=mp_ctx,
        )

        for sample in cache_dl:
            if not sample:  # all batch objects are already in cache
                continue
            for name, encoder in cache_encoders.items():
                encoder.fill_cache(sample[name])

    @classmethod
    def _switch_multiprocessing_context(cls, dataloader):
        if "PYTHONHASHSEED" in os.environ:
            return

        if dataloader.num_workers == 0:
            return

        mp_context = dataloader.multiprocessing_context
        cls._check_mp_context(mp_context)

        dataloader.multiprocessing_context = cls.CACHE_MULTIPROCESSING_CONTEXT

    @classmethod
    def _check_mp_context(cls, mp_context):
        if not mp_context:
            return

        if not isinstance(mp_context, str):
            mp_context = mp_context.get_start_method()

        if mp_context != cls.CACHE_MULTIPROCESSING_CONTEXT:
            logger.warning(
                "Default start method on your OS is not "
                f"{cls.CACHE_MULTIPROCESSING_CONTEXT}. "
                "Trying to switch it. However "
                f"{cls.CACHE_MULTIPROCESSING_CONTEXT} may be unsafe or "
                "unsupported. The most safe option is to launch your process "
                "with fixed `PYTHONHASHSEED` env and remain `spawn` as start "
                "method.\n"
                "Possible launch is `PYTHONHASHSEED=0 python3 run.py"
            )

        if cls.CACHE_MULTIPROCESSING_CONTEXT not in mp.get_all_start_methods():
            raise OSError(
                f"Cache can't be used without setting `PYTHONHASHSEED`. "
                f"{cls.CACHE_MULTIPROCESSING_CONTEXT} multiprocessing context "
                "is not available on current OS"
            )
