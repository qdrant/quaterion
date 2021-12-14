from enum import Enum


class TrainStage(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "TEST"
