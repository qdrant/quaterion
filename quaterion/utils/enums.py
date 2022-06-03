from enum import Enum


class TrainStage(str, Enum):
    """Handle train stage."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
