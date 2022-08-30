from dataclasses import dataclass
from enum import Enum
from typing import Optional


class XbmDevice(str, Enum):
    """Device selection for placement of the buffer"""

    CPU = "cpu"
    """Buffer created in CPU"""

    CUDA = "cuda"
    """Buffer created in GPU"""

    AUTO = "auto"
    """Buffer created in GPU if available. In CPU, otherwise."""


@dataclass
class XbmConfig:
    """Determine XBM settings.

    This class should be returned from
    :meth:`~quaterion.train.trainable_model.TrainableModel.configure_xbm`
    """

    weight: Optional[float] = 1.0
    """Value to scale the buffer loss before adding it to the final loss"""

    buffer_size: Optional[int] = 10000
    """Size of the memory buffer that holds embeddings from previous batches"""

    start_iteration: Optional[int] = 1000
    """Iteration step to start considering the buffer loss"""

    device: Optional[XbmDevice] = XbmDevice.AUTO
    """Placement of the buffer"""
