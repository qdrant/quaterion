import json
import random
import os
from typing import Dict, Union, List, Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from quaterion import Quaterion, TrainableModel
from quaterion.dataset.similarity_data_loader import (
    GroupSimilarityDataLoader,
    SimilarityGroupSample,
)
from quaterion.loss import SoftmaxLoss, SimilarityLoss
from quaterion_models.types import CollateFnType
from quaterion_models.encoders import Encoder
from quaterion_models.heads import GatedHead, EncoderHead
from torch.utils.data import DataLoader, Dataset

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    import sys

    print("You need to install sentence-transformers for this example")
    sys.exit(1)


random.seed(42)


class StartupsDataset(Dataset):
    def __init__(self, max_samples: int = 500):
        super().__init__()
        path: str = os.path.join(os.path.dirname(__file__), "data", "startups.jsonl")
        with open(path, "r", encoding="utf8") as f:
            lines = f.readlines()[:max_samples]
            random.shuffle(lines)
            self.data = [json.loads(line) for line in lines]

        industries = set(sorted([item["industry"] for item in self.data]))

        self._label2idx = {label: idx for idx, label in enumerate(industries)}

    def __getitem__(self, index: int) -> SimilarityGroupSample:
        item = self.data[index]
        return SimilarityGroupSample(obj=item, group=self._label2idx[item["industry"]])

    def __len__(self) -> int:
        return len(self.data)

    def get_num_industries(self) -> int:
        return len(self._label2idx)


class StartupEncoder(Encoder):
    def __init__(self, pretrained_name: str):
        super().__init__()
        self.encoder = SentenceTransformer(pretrained_name)

        self._pretrained_name = pretrained_name

    def trainable(self) -> bool:
        return False

    def embedding_size(self) -> int:
        return self.encoder.get_sentence_embedding_dimension()

    def get_collate_fn(self) -> CollateFnType:
        return self.extract_texts

    def extract_texts(self, batch: List[Union[str, Dict[str, Any]]]):
        if isinstance(batch[0], str):
            return batch
        elif isinstance(batch[0], Dict):
            return [item["description"] for item in batch]
        else:
            raise TypeError("Expecting list of strings or dicts as inputs")

    def forward(self, inputs):
        return self.encoder.encode(
            inputs, convert_to_numpy=False, convert_to_tensor=True
        )

    def save(self, output_path: str):
        self.encoder.save(os.path.join(output_path, self._pretrained_name))

    @classmethod
    def load(cls, input_path: str) -> "Encoder":
        return StartupEncoder(input_path)


class Model(TrainableModel):
    def __init__(
        self,
        pretrained_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        num_groups: int = 27,
        lr: float = 3e-5,
    ):
        self._pretrained_name = pretrained_name
        self._num_groups = num_groups
        self._lr = lr
        super().__init__()

    def configure_encoders(self) -> Union[Encoder, Dict[str, Encoder]]:
        return StartupEncoder(self._pretrained_name)

    def configure_head(self, input_embedding_size) -> EncoderHead:
        return GatedHead(input_embedding_size)

    def configure_loss(self) -> SimilarityLoss:
        return SoftmaxLoss(self.model.head.output_size(), self._num_groups)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [
                {"params": self.model.parameters(), "lr": self._lr},
                {"params": self.loss.parameters(), "lr": self._lr * 10.0},
            ]
        )

        return optimizer


dataset = StartupsDataset(max_samples=640)

model = Model(num_groups=dataset.get_num_industries(), lr=3e-5)

train_dataloader = GroupSimilarityDataLoader(dataset, batch_size=64, shuffle=True)

trainer = pl.Trainer(gpus=1, num_nodes=1, max_epochs=30)

Quaterion.fit(
    trainable_model=model,
    trainer=trainer,
    train_dataloader=train_dataloader,
)

model.save_servable("startups")
