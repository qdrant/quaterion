from typing import Dict

import torch


class Accumulator:
    def __init__(self):
        self._embeddings = []
        self._filled = False

    @property
    def state(self) -> Dict[str, torch.Tensor]:
        return {"embeddings": self.embeddings}

    @property
    def filled(self):
        return self._filled

    def set_filled(self):
        self._filled = True

    @property
    def embeddings(self):
        """Concatenate list of embeddings to Tensor

        Help to avoid concatenating embeddings for each batch during accumulation. Instead,
        concatenate it only on call.

        Returns:
            torch.Tensor: batch of embeddings
        """
        return torch.cat(self._embeddings) if len(self._embeddings) else torch.Tensor()

    def update(self, **kwargs) -> None:
        """Accumulate batch

        Args:
            **kwargs - embeddings and objects required for label calculation. E.g. for
            pair-based tasks it is `labels`, `pairs`, `subgroups` and for group-based tasks it is
            `groups`.
        """
        raise NotImplementedError()

    def reset(self):
        """Reset accumulated state

        Use to reset accumulated embeddings, labels
        """
        self._embeddings = []
        self._filled = False
