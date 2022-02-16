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
