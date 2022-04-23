import os

import torch
import torch.nn as nn
from quaterion_models.encoders import Encoder

try:
    import torchvision
except ImportError:
    import sys

    print("You need to install torchvision for this example:")
    print("pip install torchvision")

    sys.exit(1)


class CarsEncoder(Encoder):
    def __init__(self, embedding_size: int = 512, restore_path: str = None):
        super().__init__()
        if restore_path is not None:
            self._encoder = torch.load(restore_path)
        else:
            # replace the classifier layer with a newly initialized linear projection
            self._encoder = torchvision.models.resnet18(pretrained=True)
            # replace classifier layer with an identity function
            self._encoder.fc = nn.Identity()

        self._embedding_size = 512  # last dimension from the base model

    @property
    def trainable(self) -> bool:
        return False

    @property
    def embedding_size(self) -> int:
        return self._embedding_size

    def forward(self, images):
        return self._encoder.forward(images)

    def save(self, output_path: str):
        os.makedirs(output_path, exist_ok=True)
        torch.save(self._encoder, os.path.join(output_path, "encoder.pth"))

    @classmethod
    def load(cls, input_path):
        return CarsEncoder(restore_path=os.path.join(input_path, "encoder.pth"))
