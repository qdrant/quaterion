from typing import Optional

import pytorch_lightning as pl
from quaterion_models.model import DEFAULT_ENCODER_KEY

from quaterion.dataset import PairsSimilarityDataLoader
from quaterion.train.cache import CacheConfig, InMemoryCacheEncoder
from tests.model_fixtures import FakePairDataset, FakeTrainableModel


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

    cached_batch, labels = next(iter(dataloader))
    cached_data = cached_batch["data"]
    cached_meta = cached_batch["meta"]
    print("cached_batch: ", cached_data)
    print("cached_meta: ", cached_meta)

    assert len(cached_data[DEFAULT_ENCODER_KEY]) == len(cached_meta)

    # check that batch for cache contains only IDs
    assert isinstance(cached_data[DEFAULT_ENCODER_KEY], list)
    assert len(cached_data[DEFAULT_ENCODER_KEY]) == batch_size * 2
    assert isinstance(cached_data[DEFAULT_ENCODER_KEY][0], int)

    cached_result = cache_trainable_model.model.forward(cached_batch)
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
