from typing import Union, Dict, Optional

import torch
from quaterion_models.encoders import Encoder
from quaterion_models.heads import EncoderHead, GatedHead
from quaterion_models.types import TensorInterchange
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

import pytorch_lightning as pl

from quaterion import TrainableModel
from quaterion.dataset import SimilarityPairSample, PairsSimilarityDataLoader
from quaterion.loss import SimilarityLoss, ContrastiveLoss
from quaterion.train.cache import (
    CacheConfig,
    InMemoryCacheEncoder,
)
from quaterion_models.model import DEFAULT_ENCODER_KEY


class FakeEncoder(Encoder):
    def __init__(self):
        super().__init__()

        self.tensors = {
            "cheesecake".strip(): torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            "muffins   ".strip(): torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            "macaroons ".strip(): torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            "candies   ".strip(): torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            "nutella   ".strip(): torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            "lemon     ".strip(): torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            "lime      ".strip(): torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            "orange    ".strip(): torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            "grapefruit".strip(): torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            "mandarin  ".strip(): torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        }

    @property
    def trainable(self) -> bool:
        return False

    @property
    def embedding_size(self) -> int:
        return 6

    def forward(self, batch: TensorInterchange) -> Tensor:
        return torch.stack([self.tensors[word] for word in batch])

    def save(self, output_path: str):
        pass

    @classmethod
    def load(cls, input_path: str) -> "Encoder":
        return FakeEncoder()


class FakeDataset(Dataset):
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> T_co:
        return self.data[index]

    def __init__(self):
        self.data = [
            SimilarityPairSample(
                obj_a="cheesecake", obj_b="muffins", score=0.9, subgroup=10
            ),
            SimilarityPairSample(
                obj_a="cheesecake", obj_b="macaroons", score=0.8, subgroup=10
            ),
            SimilarityPairSample(
                obj_a="cheesecake", obj_b="candies", score=0.7, subgroup=10
            ),
            SimilarityPairSample(
                obj_a="cheesecake", obj_b="nutella", score=0.6, subgroup=10
            ),
            # Second query group
            SimilarityPairSample(obj_a="lemon", obj_b="lime", score=0.9, subgroup=11),
            SimilarityPairSample(obj_a="lemon", obj_b="orange", score=0.7, subgroup=11),
            SimilarityPairSample(
                obj_a="lemon", obj_b="grapefruit", score=0.6, subgroup=11
            ),
            SimilarityPairSample(
                obj_a="lemon", obj_b="mandarin", score=0.6, subgroup=11
            ),
        ]


class FakeTrainableModel(TrainableModel):
    def configure_loss(self) -> SimilarityLoss:
        return ContrastiveLoss()

    def configure_encoders(self) -> Union[Encoder, Dict[str, Encoder]]:
        return FakeEncoder()

    def configure_head(self, input_embedding_size: int) -> EncoderHead:
        return GatedHead(input_embedding_size)

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.model.parameters(), lr=0.001)


class FakeCachableTrainableModel(FakeTrainableModel):
    def configure_caches(self) -> Optional[CacheConfig]:
        return CacheConfig()


def test_cache_dataloader():
    batch_size = 3

    dataset = FakeDataset()
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
    assert len(encoder.cache) == len(dataset.data) * 2

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
