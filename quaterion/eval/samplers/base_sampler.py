from torch.utils.data import Dataset
from quaterion_models import MetricModel
from quaterion.eval.base_metric import BaseMetric


class BaseSampler:
    """Sample part of embeddings and targets to perform metric calculation on a part of the data

    Sampler allows reducing amount of time and resources to calculate a distance matrix.
    Instead of calculation of squared matrix with shape (num_embeddings, num_embeddings), it
    selects embeddings and computes matrix of a rectangle shape.

        Args:
            sample_size: amount of objects to select.

    """

    def __init__(self, sample_size=-1):
        self.sample_size = sample_size

    def sample(self, dataset: Dataset, metric: BaseMetric, model: MetricModel):
        pass

    def reset(self):
        """Reset accumulated state if any"""
        pass
