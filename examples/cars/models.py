import torch
from quaterion_models.encoders import Encoder
from quaterion_models.heads import EncoderHead, SkipConnectionHead, WideningHead
from torch import nn
from torch import Tensor
from typing import Dict, Union, Optional, Any

from quaterion import TrainableModel
from quaterion.eval.group import RetrievalRPrecision
from quaterion.loss import SimilarityLoss, TripletLoss
from quaterion.train.cache import CacheConfig, CacheType
from quaterion.utils.enums import TrainStage
from torchmetrics import MeanMetric

try:
    import torchvision
except ImportError:
    import sys

    print("You need to install torchvision for this example:")
    print("pip install torchvision")

    sys.exit(1)

from .encoders import CarsEncoder


class Model(TrainableModel):
    def __init__(self, lr: float, mining: str):
        self._lr = lr
        self._mining = mining
        super().__init__()
        self._metrics = {
            "train": MeanMetric(compute_on_step=False),
            "valid": MeanMetric(compute_on_step=False),
        }

    def configure_encoders(self) -> Union[Encoder, Dict[str, Encoder]]:
        pre_trained_encoder = torchvision.models.resnet152(pretrained=True)
        pre_trained_encoder.fc = nn.Identity()
        return CarsEncoder(pre_trained_encoder)

    def configure_head(self, input_embedding_size) -> EncoderHead:
        return SkipConnectionHead(input_embedding_size, dropout=0.1)

    def configure_loss(self) -> SimilarityLoss:
        return TripletLoss(mining=self._mining, margin=0.5)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), self._lr)
        return optimizer

    def configure_caches(self) -> Optional[CacheConfig]:
        return CacheConfig(
            cache_type=CacheType.AUTO, save_dir="./cache_dir", batch_size=32
        )

    def process_results(
        self,
        embeddings: Tensor,
        targets: Dict[str, Any],
        batch_idx: int,
        stage: TrainStage,
        **kwargs,
    ):
        metric = RetrievalRPrecision()
        metric.update(embeddings, **targets)
        batch_metric = float(metric.compute())
        self._metrics[stage[:5]].update(batch_metric)
        self.log(
            f"rrp_{stage[:5]}",
            batch_metric,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )

    def on_train_epoch_end(self) -> None:
        self.log(f"rrp_epoch_train", self._metrics["train"].compute())
        self._metrics["train"].reset()

    def on_validation_epoch_end(self) -> None:
        self.log(f"rrp_epoch_valid", self._metrics["valid"].compute())
        self._metrics["valid"].reset()
