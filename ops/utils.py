from enum import Enum

import torch
from torch import Tensor


class CastingMode(Enum):
    NONE = "none"
    ALIGN = "align"
    INPUT = "input"
    ALL = "all"


_floating_dtypes = tuple(
    value for value in vars(torch).values()
    if isinstance(value, torch.dtype) and value.is_floating_point
)
_floating_bits = {dtype: torch.finfo(dtype).bits for dtype in _floating_dtypes}
_floating_eps = {dtype: torch.finfo(dtype).eps for dtype in _floating_dtypes}


def get_floating_bits(dtype: torch.dtype) -> int:
    return _floating_bits[dtype]


def get_floating_eps(dtype: torch.dtype) -> float:
    return _floating_eps[dtype]


def convert_dtype(*args, dtype: torch.dtype):
    return tuple(x.to(dtype) if isinstance(x, Tensor) else x for x in args)


def promote_dtype(*args, dtype: torch.dtype):
    return tuple(x.to(dtype) if isinstance(x, Tensor) and get_floating_bits(x.dtype) < get_floating_bits(dtype) else x for x in args)
