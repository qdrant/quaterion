import pytest

import pytorch_lightning as pl

from quaterion.dataset.similarity_data_loader import PairsSimilarityDataLoader
from tests.model_fixtures import FakePairDataset, FakeTrainableModel


def test_ambiguous_batch_warning():
    fake_pair_dataset = FakePairDataset()
    fake_trainable_model = FakeTrainableModel()
    dataloader = PairsSimilarityDataLoader(fake_pair_dataset, batch_size=4)
    fake_trainable_model.setup_dataloader(dataloader)
    trainer = pl.Trainer(max_epochs=1, log_every_n_steps=1)
    with pytest.warns(None) as records:
        trainer.fit(fake_trainable_model, dataloader, val_dataloaders=dataloader)

    for record in list(records):
        if "Trying to infer" in str(record.message):
            assert False
