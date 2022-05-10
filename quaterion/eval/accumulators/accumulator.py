from typing import Dict

import torch


class Accumulator:
    """Accumulate calculated embeddings and corresponding targets for metrics and evaluators"""

    def __init__(self):
        self._embeddings = []
        self._filled = False

    @property
    def state(self) -> Dict[str, torch.Tensor]:
        """Accumulated state

        Returns:
            Dict[str, torch.Tensor] - dictionary with corresponding field names and accumulated
                values
        """
        return {"embeddings": self.embeddings}

    @property
    def filled(self) -> bool:
        """State of accumulator

        Returns:
            bool - represents whether accumulator can still accumulate values or it is already
                filled
        """
        return self._filled

    def set_filled(self):
        """Prevent further accumulation"""
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

        Use to reset accumulator state.
        """
        self._embeddings = []
        self._filled = False
