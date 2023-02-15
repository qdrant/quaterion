from typing import Any, Dict, List, Union

from quaterion_models.encoders import Encoder
from quaterion_models.types import CollateFnType

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    import sys

    print("You need to install sentence-transformers for this example")
    sys.exit(1)


class StartupEncoder(Encoder):
    def __init__(self, encoder_path: str):
        super().__init__()
        self.encoder = SentenceTransformer(encoder_path)

    @property
    def trainable(self) -> bool:
        return False

    @property
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
        self.encoder.save(output_path)

    @classmethod
    def load(cls, input_path: str) -> "Encoder":
        return StartupEncoder(input_path)
