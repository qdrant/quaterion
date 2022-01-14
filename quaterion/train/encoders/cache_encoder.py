from typing import Collection, Hashable

from torch import Tensor

from quaterion_models.encoder import Encoder, CollateFnType, TensorInterchange


class CacheEncoder(Encoder):
    def __init__(self, encoder: Encoder):
        if encoder.trainable():
            raise ValueError("Trainable encoder can't be cached")
        super().__init__()
        self._encoder = encoder

    def trainable(self) -> bool:
        return False

    def embedding_size(self) -> int:
        """
        :return: Size of resulting embedding
        """
        return self._encoder.embedding_size()

    def get_collate_fn(self) -> CollateFnType:
        """
        Provides function that converts raw data batch into suitable model
        input

        :return: Model input
        """
        return self._encoder.get_collate_fn()

    def forward(self, batch: TensorInterchange) -> Tensor:
        """
        Infer encoder - convert input batch to embeddings

        :param batch: processed batch
        :return: embeddings, shape: [batch_size x embedding_size]
        """
        raise NotImplementedError()

    def save(self, output_path: str):
        """
        Persist current state to the provided directory

        :param output_path:
        :return:
        """
        self._encoder.save(output_path)

    @classmethod
    def load(cls, input_path: str) -> "Encoder":
        """
        CachedEncoder classes wrap already instantiated encoders and don't
        provide loading support.
        
        :param input_path:
        :return:
        """
        raise ValueError("Cached encoder does not support loading")

    def fill_cache(
        self, data: Collection[Hashable],
    ):
        """
        Applies encoder to data and store results in cache

        :param data: collection of hashables to which encoder will be applied
        and the resulting embeddings will be stored in cache
        """
        raise NotImplementedError()

    def reset_cache(self):
        """
        Reset all stored data
        """
        raise NotImplementedError()
