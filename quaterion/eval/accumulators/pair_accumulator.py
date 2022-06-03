from typing import Dict

import torch

from quaterion.eval.accumulators import Accumulator


class PairAccumulator(Accumulator):
    """Accumulate embeddings, labels, pairs and subgroups for pair-based tasks.

    Keep track of current size to properly handle pairs.
    """

    def __init__(self):
        super().__init__()
        self._labels = []
        self._pairs = []
        self._subgroups = []
        self._accumulated_size = 0

    @property
    def state(self) -> Dict[str, torch.Tensor]:
        """Accumulated state

        Returns:
            Dict[str, torch.Tensor] - dictionary accumulates embeddings, labels, pairs, subgroups.
        """
        state = super().state
        state.update(
            {"labels": self.labels, "pairs": self.pairs, "subgroups": self._subgroups}
        )
        return state

    def update(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        pairs: torch.LongTensor,
        subgroups: torch.Tensor,
        device=None,
    ):
        """Update accumulator state.

        Move provided embeddings and groups to proper device and add to accumulated state.

        Args:
            embeddings: embeddings to accumulate
            labels: labels to distinguish similar and dissimilar objects.
            pairs: indices to determine objects of one pair
            subgroups: subgroups numbers to determine which samples can be considered negative
            device: device to store calculated embeddings and groups on.
        """
        device = device if device else embeddings.device

        embeddings = embeddings.detach().to(device)
        labels = labels.detach().to(device)
        pairs = pairs.detach().to(device)
        subgroups = subgroups.detach().to(device)

        self._embeddings.append(embeddings)
        self._labels.append(labels)
        self._pairs.append(pairs + self._accumulated_size)
        self._subgroups.append(subgroups)

        self._accumulated_size += embeddings.shape[0]

    def reset(self):
        """Reset accumulator state

        Reset accumulator status and size, accumulated embeddings, labels, pairs and subgroups
        """
        super().reset()
        self._labels = []
        self._pairs = []
        self._subgroups = []
        self._accumulated_size = 0

    @property
    def labels(self):
        """Concatenate list of labels to Tensor

        Help to avoid concatenating labels for each batch during accumulation. Instead,
        concatenate it only on call.

        Returns:
            torch.Tensor: batch of labels
        """
        return torch.cat(self._labels) if len(self._labels) else torch.Tensor()

    @property
    def subgroups(self):
        """Concatenate list of subgroups to Tensor

        Help to avoid concatenating subgroups for each batch during accumulation. Instead,
        concatenate it only on call.

        Returns:
            torch.Tensor: batch of subgroups
        """
        return torch.cat(self._subgroups) if len(self._subgroups) else torch.Tensor()

    @property
    def pairs(self) -> torch.LongTensor:
        """Concatenate list of pairs to Tensor

        Help to avoid concatenating pairs for each batch during accumulation. Instead,
        concatenate it only on call.

        Returns:
            torch.Tensor: batch of pairs
        """
        return torch.cat(self._pairs) if len(self._pairs) else torch.LongTensor()
