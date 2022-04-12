from typing import Optional

from quaterion.loss.group_loss import GroupLoss

try:
    from pytorch_metric_learning.losses import BaseMetricLossFunction
    from pytorch_metric_learning.miners import BaseMiner
except ImportError:
    import sys

    print("You need to install pytorch_mmetric_learning for this wrapper.")
    sys.exit(1)


class PytorchMetricLearningWrapper(GroupLoss):
    """Provides a simple wrapper to be able to use losses and miners from pytorch-metric-learning.

    You need to create loss (and optionally miner) instances yourself, and pass those instances
    to the constructor of this wrapper.

    Note:
        This is an experimental feature that may be subject to change, deprecation or removal.

    Note:
        See below for a quick usage example of this wrapper, but refer to the documentation of
        `pytorch-metric-learning` to learn more about individual
        `losses <https://kevinmusgrave.github.io/pytorch-metric-learning/losses>`__ and
        `miners <https://kevinmusgrave.github.io/pytorch-metric-learning/miners>`__.

    Args:
        loss: An instance of a loss object subclassing
            `pytorch_metric_learning.losses.BaseMetricLossFunction <https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#basemetriclossfunction>`__.
        miner: An instance of a miner object subclassing
            `pytorch_metric_learning.miners.BaseMetric <https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#baseminer>`__.

    Example::

        class MyTrainableModel(quaterion.TrainableModel):
            ...
            def configure_loss(self):
                loss = pytorch_metric_learning.losses.TripletMarginLoss()
                miner = pytorch_metric_learning.miner.MultiSimilarityMiner()
                return quaterion.loss.PytorchMetricLearningWrapper(loss, miner)

    """

    def __init__(self, loss: BaseMetricLossFunction, miner: Optional[BaseMiner] = None):
        super().__init__()
        self._loss = loss
        self._miner = miner

    def forward(self, embeddings, groups):
        mined_indices = None
        if self._miner is not None:
            mined_indices = self._miner(embeddings, groups)

        return self._loss(embeddings, groups, mined_indices)
