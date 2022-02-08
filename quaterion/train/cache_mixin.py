import os
import multiprocessing as mp

from typing import (
    Union,
    Dict,
    Optional,
    Set,
    Any,
    Callable,
    Hashable,
    List,
    Iterable,
    Tuple,
)

import torch.cuda
import pytorch_lightning as pl

from loguru import logger
from torch.utils.data import DataLoader
from pytorch_lightning.loops import (
    FitLoop,
    TrainingEpochLoop,
    TrainingBatchLoop,
    EvaluationLoop,
)
from pytorch_lightning.utilities.types import (
    TRAIN_DATALOADERS,
    EVAL_DATALOADERS,
)
from quaterion_models.encoders import Encoder
from quaterion_models.model import DEFAULT_ENCODER_KEY
from quaterion_models.types import TensorInterchange

from quaterion.dataset.cache_data_loader import CacheDataLoader
from quaterion.train.encoders.cache_encoder import KeyExtractorType
from quaterion.dataset.similarity_data_loader import SimilarityDataLoader
from quaterion.train.encoders import (
    CacheConfig,
    CacheEncoder,
    CacheType,
    InMemoryCacheEncoder,
)


class CacheMixin:
    CACHE_MULTIPROCESSING_CONTEXT = "fork"

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
                    encoder,
                    cache_type,
                    key_extractor,
                    encoder_name,
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
                encoders,
                cache_config.cache_type,
                key_extractor,
                encoder_name,
            )
        else:
            raise ValueError(
                "If cache is configured, cache_type or mapping have to be set"
            )

        return encoders

    @staticmethod
    def _check_cuda(cache_type: CacheType, encoder_name: str) -> None:
        """
        If encoder is supposed to use GPU as tensor's storage, then cuda
        has to be available, otherwise raises ValueError

        Args:
            cache_type: cache type for encoder
            encoder_name: name of cache encoder

        Returns:
            None

        Raises:
            ValueError: If cuda is not available
        """
        if cache_type == CacheType.GPU and not torch.cuda.is_available():
            raise ValueError(
                f"`CacheType.GPU` has been chosen for `{encoder_name}` "
                "encoder, but cuda is not available"
            )

    @staticmethod
    def _wrap_encoder(
        encoder: Encoder,
        cache_type: CacheType,
        key_extractor: Optional[KeyExtractorType],
        encoder_name: str = "",
    ) -> CacheEncoder:
        """Wrap encoder into CacheEncoder instance.

        Args:
            encoder: raw model's encoder
            cache_type: cache type of tensor storage
            key_extractor: function to obtain hashable values for complex
                object which can't be hashed with standard means
            encoder_name: name of encoder to be wrapped

        Returns:
            CacheEncoder: wrapped encoder

        Raises:
            ValueError: if encoder layers are not frozen. Cache can be applied
                only to fully frozen encoders' outputs.
        """
        if encoder.trainable():
            raise ValueError(
                f"Can't configure cache for encoder {encoder_name}. "
                "Encoder must be frozen to cache it"
            )

        encoder = InMemoryCacheEncoder(encoder, key_extractor, cache_type)

        return encoder

    @classmethod
    def cache(
        cls,
        trainer: pl.Trainer,
        encoders: Dict[str, Encoder],
        train_dataloader: SimilarityDataLoader,
        val_dataloader: Optional[SimilarityDataLoader],
        cache_config: CacheConfig,
    ) -> None:
        """Filling cache for model's cache encoders.

        Args:
            trainer: performs one training epoch to cache encoders outputs.
                Preserve all encapsulated pytorch-lightning logic such as
                device managing etc.
            encoders: mapping of model's encoders and their names
            train_dataloader: model's train dataloader
            val_dataloader: model's val dataloader
            cache_config: cache config instance to configure cache batch size
                and num of workers to use for caching

        Returns:
            None
        """
        cache_encoders = {
            name: encoder
            for name, encoder in encoders.items()
            if isinstance(encoder, CacheEncoder)
        }

        if not cache_encoders:
            return

        cls._compatibility_check(train_dataloader)
        train_dataloader = cls._wrap_cache_dataloader(
            train_dataloader, cache_config, cache_encoders
        )

        if val_dataloader is not None:
            val_dataloader = cls._wrap_cache_dataloader(
                val_dataloader, cache_config, cache_encoders
            )

        cls._fill_cache(trainer, cache_encoders, train_dataloader, val_dataloader)

        # Once cache is filled, collate functions return only keys for cache
        for encoder in cache_encoders.values():
            encoder.cache_filled = True
        logger.info("Caching has been successfully finished")

    @classmethod
    def _fill_cache(
        cls,
        trainer: pl.Trainer,
        cache_encoders: Dict[str, CacheEncoder],
        train_dataloader: SimilarityDataLoader,
        val_dataloader: SimilarityDataLoader,
    ) -> None:
        """Fills cache and restores trainer state for further training process.

        Args:
            trainer: performs one training and validation epoch
            cache_encoders: mapping of encoders to cache input
            train_dataloader: model's train dataloader
            val_dataloader: model's val dataloader

        Returns:
            None
        """
        # store configured fit and validate loops to restore them for training
        # process after cache
        _fit_loop = trainer.fit_loop
        _val_loop = trainer.validate_loop

        # Mimic fit loop configuration from trainer
        fit_loop = FitLoop(
            min_epochs=1,
            max_epochs=1,
        )
        training_epoch_loop = TrainingEpochLoop()
        training_batch_loop = TrainingBatchLoop()
        training_validation_loop = EvaluationLoop()
        training_epoch_loop.connect(
            batch_loop=training_batch_loop, val_loop=training_validation_loop
        )
        fit_loop.connect(epoch_loop=training_epoch_loop)
        trainer.fit_loop = fit_loop

        # The actual caching
        trainer.predict(
            CacheModel(
                cache_encoders,
            ),
            [train_dataloader, val_dataloader],
        )
        trainer.fit_loop = _fit_loop

    @classmethod
    def _compatibility_check(cls, dataloader: DataLoader) -> None:
        """Checks whether dataloader type.

        To be used in caching, dataloader has to have an implementation of
        `cache_collate_fn`. Currently, only `SimilarityDataLoader` has it.

        Args:
            dataloader: DataLoader

        Returns:
            None

        Raises:
            TypeError: if dataloader is not instance of SimilarityDataLoader
        """
        if not isinstance(dataloader, SimilarityDataLoader):
            raise TypeError("DataLoader must be SimilarityDataLoader ")

    @classmethod
    def _wrap_cache_dataloader(
        cls,
        dataloader: SimilarityDataLoader,
        cache_config: CacheConfig,
        cache_encoders: Dict[str, CacheEncoder],
    ) -> CacheDataLoader:
        """Wrap dataloader for caching.

        Child processes need to derive randomized `PYTHONHASHSEED` value from
        parent process to obtain the same hash values. It is only done with
        `fork` start method.
        `fork` start method is presented on Unix systems, and it is the default
        for them, except macOS. Therefore, we set `fork` method explicitly.

        If dataloader is not supposed to use child process, nothing being done.
        If multiprocessing_context was set by user, it is being untouched and
        can lead to errors.
        If `PYTHONHASHSEED` is set explicitly then multiprocessing_context
        won't be switched.
        Cache can be used on Windows only in case when `PYTHONHASHSEED` is set
        explicitly.

        Args:
            dataloader: dataloader to be wrapped
            cache_config: cache config to retrieve num of workers and batch
                size
            cache_encoders: encoders to set key extractors and collate_fns

        Returns:
            CacheDataLoader: wrapped dataloader
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

        # We need to reduce random sampling and repeated calculations to
        # make cache as fast as possible. Thus, we recreate dataloader
        # and set batch size explicitly.
        cache_dl = CacheDataLoader(
            key_extractors=dict(
                (encoder_name, cache_encoder.key_extractor)
                for encoder_name, cache_encoder in cache_encoders.items()
            ),
            cached_encoders_collate_fns=dict(
                (encoder_name, cache_encoder.get_collate_fn())
                for encoder_name, cache_encoder in cache_encoders.items()
            ),
            unique_objects_extractor=dataloader.fetch_unique_objects,
            dataset=dataloader.dataset,
            batch_size=cache_config.batch_size,
            num_workers=num_workers,
            pin_memory=dataloader.pin_memory,
            timeout=dataloader.timeout,
            worker_init_fn=dataloader.worker_init_fn,
            prefetch_factor=dataloader.prefetch_factor,
            multiprocessing_context=mp_ctx,
        )
        return cache_dl

    @classmethod
    def _switch_multiprocessing_context(cls, dataloader: SimilarityDataLoader) -> None:
        """Switch dataloader multiprocessing context.

        Do nothing if dataloader is not supposed to use child processes or
        `PYTHONHASHSEED` has been set explicitly by user.

        Args:
            dataloader: dataloader to check and switch multiprocessing context

        Returns:
            None
        """
        if "PYTHONHASHSEED" in os.environ:
            return

        if dataloader.num_workers == 0:
            return

        mp_context: Optional[
            Union[str, mp.context.BaseContext]
        ] = dataloader.multiprocessing_context
        cls._check_mp_context(mp_context)

        dataloader.multiprocessing_context = cls.CACHE_MULTIPROCESSING_CONTEXT

    @classmethod
    def _check_mp_context(
        cls, mp_context: Optional[Union[str, mp.context.BaseContext]]
    ) -> None:
        """Check if multiprocessing context is compatible with cache.

        Emits warning if current multiprocessing context start method does not
        coincide with one required by cache.

        Args:
            mp_context: some dataloader's multiprocessing context

        Returns:
            None

        Raises:
            OSError: Raise OSError if OS does not support process start method
            required by cache. Currently, this start method is `fork` which is
            not supported by Windows.
        """
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


class CacheModel(pl.LightningModule):
    """Mock model for convenient caching.

    This class is required to make caching process similar to the training of
    the genuine model and inherit and use the same trainer instance. It allows
    avoiding of messing with device managing stuff and more.
    """

    def __init__(
        self,
        encoders: Dict[str, CacheEncoder],
    ):
        super().__init__()
        self.encoders = encoders
        for key, encoder in self.encoders.items():
            self.add_module(key, encoder)

    def predict_step(
        self,
        batch: Dict[str, Tuple[Iterable[Hashable], TensorInterchange]],
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ):
        """Caches batch of input.

        Args:
            batch: batch of collated data. Contains mapping, where key is
                encoder's name, value is tuple of key used in cache and
                according items, which are ready to be processed by specific
                encoder.
            batch_idx: Index of current batch
            dataloader_idx: Index of the current dataloader
        Returns:
            torch.Tensor: loss mock
        """
        for encoder_name, encoder in self.encoders.items():
            encoder_samples = batch.get(encoder_name)
            if not encoder_samples:
                continue
            encoder.fill_cache(encoder_samples)

        return torch.Tensor([1])

    # region anchors
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        pass

    def test_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def val_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass

    # endregion
