from typing import Iterable, Optional, Sized, Union

import torch
import torch.nn.functional as F
import tqdm
from torch.utils.data import Dataset


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


def get_anchor_positive_mask(
    labels_a: torch.Tensor, labels_b: Optional[torch.Tensor] = None
) -> torch.BoolTensor:
    """Creates a 2D mask of valid anchor-positive pairs.

    Args:
        labels_a (torch.Tensor): Labels associated with embeddings in the batch A. Shape: (batch_size_a,)
        labels_b (torch.Tensor): Labels associated with embeddings in the batch B. Shape: (batch_size_b,)
        If `labels_b is None`, it assigns `labels_a` to `labels_b`.

    Returns:
        torch.Tensor: Anchor-positive mask. Shape: (batch_size_a, batch_size_b)
    """
    if labels_b is None:
        labels_b = labels_a

    # Shape: (batch_size_a, batch_size_b)
    mask = labels_a.expand(labels_b.shape[0], labels_a.shape[0]).t() == labels_b.expand(
        labels_a.shape[0], labels_b.shape[0]
    )

    if torch.equal(
        labels_a, labels_b
    ):  # handle identical batches of labels for regular loss
        # shape: (batch_size_a, batch_size_a)
        indices_equal = torch.eye(
            labels_a.size()[0], dtype=torch.bool, device=labels_a.device
        )
        indices_not_equal = torch.logical_not(indices_equal)
        mask = torch.logical_and(indices_not_equal, mask)

    return mask


def get_anchor_negative_mask(
    labels_a: torch.Tensor, labels_b: Optional[torch.Tensor] = None
) -> torch.BoolTensor:
    """Creates a 2D mask of valid anchor-negative pairs.

    Args:
        labels_a (torch.Tensor): Labels associated with embeddings in the batch A. Shape: (batch_size_a,)
        labels_b (torch.Tensor): Labels associated with embeddings in the batch B. Shape: (batch_size_b,).
        If `labels_b is None`, it assigns `labels_a` to `labels_b`.

    Returns:
        torch.Tensor: Anchor-negative mask. Shape: (batch_size_a, batch_size_b)
    """
    if labels_b is None:
        labels_b = labels_a

    # Shape: (batch_size_a, batch_size_b)
    mask = labels_a.expand(labels_b.shape[0], labels_a.shape[0]).t() != labels_b.expand(
        labels_a.shape[0], labels_b.shape[0]
    )

    return mask


def iter_by_batch(
    sequence: Union[Sized, Iterable, Dataset],
    batch_size: int,
    log_progress: bool = True,
):
    """Iterate through index-able or iterable by batches

    Try to iterate by indices, if fail - via iterable interface.
    """

    try:
        sequence.__getitem__(0)
        size = len(sequence)
        step = batch_size if batch_size < size else size
        if log_progress:
            iterator = tqdm.tqdm(range(0, size, step), total=size / step)
        else:
            iterator = range(0, size, step)

        for slice_start_index in iterator:
            slice_end_index = slice_start_index + step
            slice_end_index = slice_end_index if slice_end_index < size else size
            input_batch = [
                sequence[index] for index in range(slice_start_index, slice_end_index)
            ]
            yield input_batch

    except (AttributeError, NotImplementedError, IndexError):
        batch = []
        if log_progress:
            iterator = tqdm.tqdm(sequence)
        else:
            iterator = sequence
        for item in iterator:
            batch.append(item)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if len(batch) > 0:
            yield batch
        return


def get_masked_maximum(
    dists: torch.Tensor, mask: torch.Tensor, dim: int = 1
) -> torch.Tensor:
    """Utility function for semi hard mining.

    Args:
        dists: Tiled distance matrix.
        mask: Tiled mask.
        dim: Dimension to operate on.

    Returns:
        torch.Tensor - masked maximums.
    """
    axis_minimums, _ = dists.min(dim, keepdims=True)
    masked_maximums = (dists - axis_minimums) * mask
    masked_maximums, _ = masked_maximums.max(dim, keepdims=True)
    masked_maximums += axis_minimums

    return masked_maximums


def get_masked_minimum(dists, mask, dim=1):
    """Utility function for semi hard mining.

    Args:
        dists: Tiled distance matrix.
        mask: Tiled mask.
        dim: Dimension to operate on.

    Returns:
        torch.Tensor - masked maximums.
    """
    axis_maximums, _ = dists.max(dim, keepdims=True)
    masked_minimums = (dists - axis_maximums) * mask
    masked_minimums, _ = masked_minimums.min(dim, keepdims=True)
    masked_minimums += axis_maximums

    return masked_minimums


def l2_norm(inputs: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Apply L2 normalization to tensor

    Args:
        inputs: Input tensor.
        dim: Dimension to operate on.

    Returns:
        torch.Tensor: L2-normalized tensor
    """
    return F.normalize(inputs, p=2, dim=dim)
