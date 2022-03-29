from typing import Union, Dict, Optional, Callable

import torch
from torch import Tensor

from quaterion_models.encoders.encoder import Encoder


class BaseMetric:
    """Base metric class

    Perform calculation and accumulating of embeddings and labels from batches.

    Args:
        encoder: :class:`~quaterion_models.encoders.encoder.Encoder` instance to calculate
            embeddings.
        distance_metric: function for distance matrix computation. Possible choice might be one of
            :class:`~quaterion.loss.metrics.SiameseDistanceMetric` methods.

    """
    def __init__(self, encoder: Encoder, distance_metric: Callable):
        self.encoder = encoder
        self.distance_metric = distance_metric

        self.embeddings = Tensor()
        self.labels: Optional[Union[Tensor, Dict]] = None

    def compute(self) -> Tensor:
        """Calculates metric

        Returns:
            Tensor: metric result
        """
        raise NotImplementedError()

    def calculate_distances(self) -> Tensor:
        """Calculates distance matrix

        Returns:
            Tensor: distance matrix
        """
        return self.distance_metric(self.embeddings, self.embeddings, matrix=True)

    def update(
        self, features: Tensor, labels: Union[Tensor, Dict], device="cpu"
    ) -> None:
        """Process and accumulate batch

        Args:
            features: ready encoder input.
            labels: labels to be accumulated with corresponding embeddings.
            device: device to store calculated embeddings and labels on.
        """
        training = self.encoder.training
        self.encoder.eval()
        with torch.inference_mode():
            embeddings = self.encoder(features).to(device)
        if training:
            self.encoder.train()

        self.embeddings = torch.cat([self.embeddings, embeddings])

        if isinstance(labels, dict):
            labels = {key: value.to(device) for key, value in labels.items()}
        else:
            labels = labels.to(device)

        if self.labels is None:
            self.labels = labels
        else:
            if isinstance(labels, dict):
                for key, value in labels.items():
                    self.labels[key] = torch.cat([value, labels[key]])
            else:
                self.labels = torch.cat([self.labels, labels])
