from typing import Dict

import torch

from quaterion.eval.accumulators import Accumulator


class GroupAccumulator(Accumulator):
    def __init__(self):
        super().__init__()
        self._groups = []

    @property
    def state(self) -> Dict[str, torch.Tensor]:
        state = super().state
        state.update({"groups": self.groups})
        return state

    def update(self, embeddings: torch.Tensor, groups: torch.Tensor, device=None):
        if device is None:
            device = embeddings.device

        embeddings = embeddings.detach().to(device)
        groups = groups.detach().to(device)

        self._embeddings.append(embeddings)
        self._groups.append(groups)

    def reset(self):
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
