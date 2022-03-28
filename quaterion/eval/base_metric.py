from typing import Union, Dict

import torch
from torch import Tensor


class BaseMetric:
    def __init__(self, encoder, distance_metric):
        self.encoder = encoder
        self.distance_metric = distance_metric

        self.embeddings = Tensor()
        self.labels = None

    def compute(self):
        raise NotImplementedError()

    def calculate_distances(self):
        return self.distance_metric(self.embeddings, self.embeddings, matrix=True)

    def update(
        self, features: Tensor, labels: Union[Tensor, Dict], device="cpu"
    ) -> None:
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
                    self.labels[key] = torch.cat(value, labels[key])
            else:
                self.labels = torch.cat([self.labels, labels])
