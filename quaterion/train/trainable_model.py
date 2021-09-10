import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT

from quaterion.loss.similarity_loss import SimilarityLoss
from quaterion_models.model import MetricModel


class TrainableModel(pl.LightningModule):

    def __init__(self,
                 model: MetricModel,
                 loss: SimilarityLoss,
                 ):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx, **kwargs) -> STEP_OUTPUT:
        pass
