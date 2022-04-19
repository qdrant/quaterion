import os

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
        if restore_path is not None:
            self._encoder = torch.load(restore_path)
        else:
            self._encoder = torchvision.models.mobilenet_v3_small(pretrained=True)
            self._encoder.classifier = nn.Linear(576, embedding_size)

        self._embedding_size = self._encoder.classifier.out_features

    @property
    def trainable(self) -> bool:
        return True

    @property
    def embedding_size(self) -> int:
        return self.embedding_size

    def save(self, output_path: str):
        os.makedirs(output_path, exist_ok=True)
        torch.save(self._encoder, os.path.join(output_path, "encoder.pth"))

    @classmethod
    def load(cls, input_path) -> CarsEncoder:
        return CarsEncoder(restore_path=os.path.join(input_path, "encoder.pth"))



