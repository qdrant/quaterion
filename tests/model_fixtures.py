from typing import Any, Dict, List, Union

import torch
from quaterion_models.encoders import Encoder
from quaterion_models.heads import EncoderHead, GatedHead
from quaterion_models.types import TensorInterchange
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

from quaterion import TrainableModel
from quaterion.dataset import SimilarityPairSample
from quaterion.loss import ContrastiveLoss, SimilarityLoss


class FakeEncoder(Encoder):
    def __init__(self):
        super().__init__()

        self.tensors = {
            "cheesecake".strip(): torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            "muffins   ".strip(): torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            "macaroons ".strip(): torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            "candies   ".strip(): torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            "nutella   ".strip(): torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            "lemon     ".strip(): torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            "lime      ".strip(): torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            "orange    ".strip(): torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            "grapefruit".strip(): torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            "mandarin  ".strip(): torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        }

    @classmethod
    def extract_meta(cls, batch: List[Any]) -> List[dict]:
        return [
            {
                "first_letter": name[0],
            }
            for name in batch
        ]

    @property
    def trainable(self) -> bool:
        return False

    @property
    def embedding_size(self) -> int:
        return 6

    def forward(self, batch: TensorInterchange) -> Tensor:
        return torch.stack([self.tensors[word] for word in batch])

    def save(self, output_path: str):
        pass

    @classmethod
    def load(cls, input_path: str) -> "Encoder":
        return FakeEncoder()


class FakePairDataset(Dataset):
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> T_co:
        return self.data[index]

    def __init__(self):
        self.data = [
            SimilarityPairSample(
                obj_a="cheesecake", obj_b="muffins", score=0.9, subgroup=10
            ),
            SimilarityPairSample(
                obj_a="cheesecake", obj_b="macaroons", score=0.8, subgroup=10
            ),
            SimilarityPairSample(
                obj_a="cheesecake", obj_b="candies", score=0.7, subgroup=10
            ),
            SimilarityPairSample(
                obj_a="cheesecake", obj_b="nutella", score=0.6, subgroup=10
            ),
            # Second query group
            SimilarityPairSample(obj_a="lemon", obj_b="lime", score=0.9, subgroup=11),
            SimilarityPairSample(obj_a="lemon", obj_b="orange", score=0.7, subgroup=11),
            SimilarityPairSample(
                obj_a="lemon", obj_b="grapefruit", score=0.6, subgroup=11
            ),
            SimilarityPairSample(
                obj_a="lemon", obj_b="mandarin", score=0.6, subgroup=11
            ),
        ]


class FakeTrainableModel(TrainableModel):
    def configure_loss(self) -> SimilarityLoss:
        return ContrastiveLoss()

    def configure_encoders(self) -> Union[Encoder, Dict[str, Encoder]]:
        return FakeEncoder()

    def configure_head(self, input_embedding_size: int) -> EncoderHead:
        return GatedHead(input_embedding_size)

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.model.parameters(), lr=0.001)
