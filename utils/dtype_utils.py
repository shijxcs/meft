from enum import Enum
from typing import *

import torch
from torch import Tensor

torch.compiler.allow_in_graph(torch.finfo)


class CastingMode(Enum):
    NONE = "none"
    ALIGN = "align"
    INPUT = "input"
    ALL = "all"


def get_float_bits(dtype: torch.dtype):
    return torch.finfo(dtype).bits


def get_float_eps(dtype: torch.dtype):
    return torch.finfo(dtype).eps


def convert_dtype(*args, dtype=None):
    return tuple(x.to(dtype) if isinstance(x, Tensor) else x for x in args)


def promote_dtype(*args, dtype=None):
    return tuple(x.to(dtype) if isinstance(x, Tensor) and get_float_bits(x.dtype) < get_float_bits(dtype) else x for x in args)
