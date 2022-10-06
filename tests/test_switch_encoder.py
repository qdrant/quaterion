import pytorch_lightning as pl
from quaterion import Quaterion
from quaterion.dataset import PairsSimilarityDataLoader

from .model_fixtures import FakeTrainableModelWithSwitchEncoder, FakePairDataset


class TestSwitchEncoder:
    def test_switch_encoder_and_head(self):
        model = FakeTrainableModelWithSwitchEncoder()
        dataset = FakePairDataset()
        data_loader = PairsSimilarityDataLoader(dataset, batch_size=3)
        trainer_args = Quaterion.trainer_defaults(model, data_loader)
        trainer_args["callbacks"].pop(1)  # remove EarlyStopping callback
        trainer_args["accelerator"] = "cpu"
        trainer_args["max_epochs"] = 1
        Quaterion.fit(
            trainable_model=model,
            trainer=pl.Trainer(**trainer_args),
            train_dataloader=data_loader,
        )
