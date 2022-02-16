from enum import Enum


class TrainStage(str, Enum):
    """Enum to handle train stage."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
