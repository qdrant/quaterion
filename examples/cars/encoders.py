import os

import torch
import torch.nn as nn
from quaterion_models.encoders import Encoder


class CarsEncoder(Encoder):
    def __init__(self, encoder_model: nn.Module):
        super().__init__()
        self._encoder = encoder_model
        self._embedding_size = 2048  # last dimension from the ResNet model

    @property
    def trainable(self) -> bool:
        return False

    @property
    def embedding_size(self) -> int:
        return self._embedding_size

    def forward(self, images):
        embeddins = self._encoder.forward(images)
        return embeddins

    def save(self, output_path: str):
        os.makedirs(output_path, exist_ok=True)
        torch.save(self._encoder, os.path.join(output_path, "encoder.pth"))

    @classmethod
    def load(cls, input_path):
        encoder_model = torch.load(os.path.join(input_path, "encoder.pth"))
        return CarsEncoder(encoder_model)
