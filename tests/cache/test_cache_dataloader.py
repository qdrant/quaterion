from typing import Optional

import pytorch_lightning as pl
from quaterion.dataset import PairsSimilarityDataLoader
from quaterion.train.cache import (
    CacheConfig,
    InMemoryCacheEncoder,
)
from quaterion_models.model import DEFAULT_ENCODER_KEY
from tests.model_fixtures import FakeTrainableModel, FakePairDataset


class FakeCachableTrainableModel(FakeTrainableModel):
    def configure_caches(self) -> Optional[CacheConfig]:
        return CacheConfig()


def test_cache_dataloader():
    batch_size = 3

    dataset = FakePairDataset()
    dataloader = PairsSimilarityDataLoader(dataset, batch_size=batch_size)

    batch = next(iter(dataloader))

    ids, features, labels = batch

    print("")
    print("ids: ", ids)
    print("features: ", features)
    print("labels: ", labels)

    trainer = pl.Trainer(logger=False, gpus=None)

    cache_trainable_model = FakeCachableTrainableModel()
    cache_trainable_model.setup_cache(
        trainer=trainer, train_dataloader=dataloader, val_dataloader=None
    )

    encoder = cache_trainable_model.model.encoders[DEFAULT_ENCODER_KEY]

    assert isinstance(encoder, InMemoryCacheEncoder)
    assert len(encoder._cache) == len(dataset.data) * 2

    cached_ids, labels = next(iter(dataloader))
    print("cached_batch: ", cached_ids)

    # check that batch for cache contains only IDs
    assert isinstance(cached_ids[DEFAULT_ENCODER_KEY], list)
    assert len(cached_ids[DEFAULT_ENCODER_KEY]) == batch_size * 2
    assert isinstance(cached_ids[DEFAULT_ENCODER_KEY][0], int)

    cached_result = cache_trainable_model.model.forward(cached_ids)
    print("cached_result: ", cached_result)

    # Same, without cache
    dataloader = PairsSimilarityDataLoader(dataset, batch_size=batch_size)

    trainable_model = FakeTrainableModel()
    trainable_model.setup_dataloader(dataloader)

    features, labels = next(iter(dataloader))

    reference_result = trainable_model.model.forward(features)

    print("reference_result: ", reference_result)

    diff_result = (cached_result - reference_result).sum()

    assert abs(diff_result) < 0.0001
