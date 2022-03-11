from typing import Union

import torch


def info_value_of_dtype(dtype: torch.dtype) -> Union[torch.finfo, torch.iinfo]:
    """Returns the `finfo` or `iinfo` object of a given PyTorch data type.

    Does not allow torch.bool.

    Args:
        dtype: dtype for which to return info value

    Returns:
        Union[torch.finfo, torch.iinfo]: info about given data type

    Raises:
        TypeError: if torch.bool is passed
    """
    if dtype == torch.bool:
        raise TypeError("Does not support torch.bool")
    elif dtype.is_floating_point:
        return torch.finfo(dtype)
    else:
        return torch.iinfo(dtype)


def min_value_of_dtype(dtype: torch.dtype) -> Union[int, float]:
    """Returns the minimum value of a given PyTorch data type.

    Does not allow torch.bool.

    Args:
        dtype: dtype for which to return min value

    Returns:
        Union[int, float]: max value of dtype
    """
    return info_value_of_dtype(dtype).min


def max_value_of_dtype(dtype: torch.dtype) -> float:
    """Returns the maximum value of a given PyTorch data type.

    Does not allow torch.bool.

    Args:
        dtype: dtype for which to return max value

    Returns:
        Union[int, float]: max value of dtype
    """
    return info_value_of_dtype(dtype).max


def get_triplet_mask(labels: torch.Tensor) -> torch.Tensor:
    """Creates a 3D mask of valid triplets for the batch-all strategy.

    Given a batch of labels with `shape = (batch_size,)`
    the number of possible triplets that can be formed is:
    batch_size^3, i.e. cube of batch_size,
    which can be represented as a tensor with `shape = (batch_size, batch_size, batch_size)`.
    However, a triplet is valid if:
    `labels[i] == labels[j] and labels[i] != labels[k]`
    and `i`, `j` and `k` are distinct indices.
    This function calculates a mask indicating which ones of all the possible triplets
    are actually valid triplets based on the given criteria above.

    Args:
        labels (torch.Tensor): Labels associated with embeddings in the batch. Shape: (batch_size,)

    Returns:
        torch.Tensor: Triplet mask. Shape: (batch_size, batch_size,  batch_size)
    """
    # get a mask for distinct indices
    # Shape: (batch_size, batch_size)
    indices_equal = torch.eye(labels.size()[0], dtype=torch.bool, device=labels.device)
    indices_not_equal = torch.logical_not(indices_equal)

    # Shape: (batch_size, batch_size, 1)
    i_not_equal_j = indices_not_equal.unsqueeze(2)
    # Shape: (batch_size, 1, batch_size)
    i_not_equal_k = indices_not_equal.unsqueeze(1)
    # Shape: (1, batch_size, batch_size)
    j_not_equal_k = indices_not_equal.unsqueeze(0)

    # Shape: (batch_size, batch_size, batch_size)
    distinct_indices = torch.logical_and(
        torch.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k
    )

    # get a mask for:
    # labels[i] == labels[j] and labels[i] != labels[k]
    # Shape: (batch_size, batch_size)
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    # Shape: (batch_size, batch_size, 1)
    i_equal_j = labels_equal.unsqueeze(2)
    # Shape: (batch_size, 1, batch_size)
    i_equal_k = labels_equal.unsqueeze(1)
    # Shape: (batch_size, batch_size, batch_size)
    valid_indices = torch.logical_and(i_equal_j, torch.logical_not(i_equal_k))

    # combine masks
    mask = torch.logical_and(distinct_indices, valid_indices)

    return mask


def get_anchor_positive_mask(labels: torch.Tensor) -> torch.Tensor:
    """Creates a 2D mask of valid anchor-positive pairs.

    Args:
        labels (torch.Tensor): Labels associated with embeddings in the batch. Shape: (batch_size,)

    Returns:
        torch.Tensor: Anchor-positive mask. Shape: (batch_size, batch_size)
    """
    # get a mask for distinct i and j indices
    # Shape: (batch_size, batch_size)
    indices_equal = torch.eye(labels.size()[0], dtype=torch.bool, device=labels.device)
    indices_not_equal = torch.logical_not(indices_equal)

    # get a mask for labels[i] == labels[j]
    # Shape: (batch_size, batch_size)
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)

    # combine masks
    mask = torch.logical_and(indices_not_equal, labels_equal)

    return mask


def get_anchor_negative_mask(labels: torch.Tensor) -> torch.Tensor:
    """Creates a 2D mask of valid anchor-negative pairs.

    Args:
        labels (torch.Tensor): Labels associated with embeddings in the batch. Shape: (batch_size,)

    Returns:
        torch.Tensor: Anchor-negative mask. Shape: (batch_size, batch_size)
    """
    # get a mask for labels[i] != labels[k]
    # Shape: (batch_size, batch_size)
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    mask = torch.logical_not(labels_equal)

    return mask
