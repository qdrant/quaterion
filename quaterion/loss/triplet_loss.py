import torch
import torch.nn as nn
import torch.nn.functional as F
from quaterion.loss.group_loss import GroupLoss


class TripletLoss(GroupLoss):
    """_summary_

    Args:
        GroupLoss (_type_): _description_
    """
    def __init__(self, margin: float = 0.5, p: int = 2, mining: str = "all"):
        super(TripletLoss, self).__init__()
