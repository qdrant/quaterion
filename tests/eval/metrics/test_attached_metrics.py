from typing import Union, Dict, List

import pytest
from quaterion_models.types import TensorInterchange
from torch import Tensor

from quaterion.eval.attached_metric import AttachedMetric
from quaterion_models.encoders import Encoder
from quaterion_models.heads import EncoderHead

from quaterion.eval.pair import RetrievalPrecision, RetrievalReciprocalRank
from quaterion.loss import SimilarityLoss
from quaterion.train.trainable_model import TrainableModel


class DummyEncoder(Encoder):
    @property
    def trainable(self) -> bool:
        return False

    @property
    def embedding_size(self) -> int:
        return 42

    def forward(self, batch: TensorInterchange) -> Tensor:
        pass

    def save(self, output_path: str):
        pass

    @classmethod
    def load(cls, input_path: str) -> Encoder:
        pass


class DummyModel(TrainableModel):
    def configure_optimizers(self):
        pass

    def configure_encoders(self) -> Union[Encoder, Dict[str, Encoder]]:
        return DummyEncoder()

    def configure_loss(self) -> SimilarityLoss:
        pass

    def configure_head(self, input_embedding_size: int) -> EncoderHead:
        pass


def test_attach():
    class NoMetricsDummyModel(DummyModel):
        pass

    assert len(NoMetricsDummyModel().attached_metrics) == 0

    class OneMetricDummyModel(DummyModel):
        def configure_metrics(self) -> Union[AttachedMetric, List[AttachedMetric]]:
            return AttachedMetric(
                "DummyMetric",
                RetrievalPrecision(),
            )

    assert len(OneMetricDummyModel().attached_metrics) == 1

    class TwoMetricsDummyModel(DummyModel):
        def configure_metrics(self) -> Union[AttachedMetric, List[AttachedMetric]]:
            return [
                AttachedMetric(
                    "DummyMetric_1",
                    RetrievalPrecision(),
                ),
                AttachedMetric(
                    "DummyMetric_2",
                    RetrievalReciprocalRank(),
                ),
            ]

    assert len(TwoMetricsDummyModel().attached_metrics) == 2


def test_attached_metric():
    metric = AttachedMetric(
        "DummyMetric_2",
        RetrievalPrecision(),
    )
    assert len(metric.stages) == 2

    metric = AttachedMetric(
        "DummyMetric_3",
        RetrievalPrecision(),
        on_prog_bar=True,
        on_epoch=False,
        on_step=True,
    )

    exp_log_options = {"on_prog_bar": True, "on_epoch": False, "on_step": True}
    assert all(
        metric.log_options[key] == value for key, value in exp_log_options.items()
    )

    metric = AttachedMetric("DummyMetric_4", RetrievalReciprocalRank())
    getattr(metric, "update")
    getattr(metric, "reset")
    getattr(metric, "compute")

    with pytest.raises(AttributeError):
        getattr(metric, "non_existent_method")
