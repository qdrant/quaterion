from typing import Dict

import torch

from quaterion.eval.accumulators import Accumulator


class GroupAccumulator(Accumulator):
    """Accumulate embeddings and groups for group-based tasks."""

    def __init__(self):
        super().__init__()
        self._groups = []

    @property
    def state(self) -> Dict[str, torch.Tensor]:
        """Accumulated state

        Returns:
            Dict[str, torch.Tensor] - dictionary with embeddings and groups.
        """
        state = super().state
        state.update({"groups": self.groups})
        return state

    def update(self, embeddings: torch.Tensor, groups: torch.Tensor, device=None):
        """Update accumulator state.

        Move provided embeddings and groups to proper device and add to accumulated state.

        Args:
            embeddings: embeddings to accumulate
            groups: corresponding groups to accumulate
            device: device to store tensors on
        """
        if device is None:
            device = embeddings.device

        embeddings = embeddings.detach().to(device)
        groups = groups.detach().to(device)

        self._embeddings.append(embeddings)
        self._groups.append(groups)

    def reset(self):
        """Reset accumulator state

        Reset accumulator status, accumulated embeddings and groups
        """
        super().reset()
        self._groups = []

    @property
    def groups(self):
        """Concatenate list of groups to Tensor

        Help to avoid concatenating groups for each batch during accumulation. Instead,
        concatenate it only on call.

        Returns:
            torch.Tensor: batch of groups
        """
        return torch.cat(self._groups) if len(self._groups) else torch.Tensor()
