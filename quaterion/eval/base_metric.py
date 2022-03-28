from typing import List

import torch


class BaseMetric:
    def __init__(self, encoder, distance_metric):
        self.encoder = encoder
        self.distance_metric = distance_metric

        self.embeddings = None
        self.labels = None

    def compute(self):
        raise NotImplementedError()

    def calculate_distances(self):
        return self.distance_metric(self.embeddings, self.embeddings, matrix=True)

    def update(self, batch: List) -> None:
        _, features, labels = batch
        embeddings = self.encoder(features).detach().to("cpu")
        self.embeddings = (
            embeddings
            if self.embeddings is None
            else torch.cat([self.embeddings, embeddings])
        )
        if self.labels is not None:
            if isinstance(labels, dict):
                for key in labels:
                    self.labels[key] = torch.cat(
                        [self.labels[key].to("cpu"), labels[key]]
                    )
            else:
                self.labels = torch.cat([self.labels.to("cpu"), labels.to("cpu")])
        else:
            self.labels = labels
