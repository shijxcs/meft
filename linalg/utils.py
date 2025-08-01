from math import sqrt

import torch
from torch import Tensor

from . import config


def scaled_matmul(input: Tensor, other: Tensor) -> Tensor:
    if config.SCALING_UNIT:
        scale = 1 / sqrt(input.shape[-1])
        return (input * scale) @ (other * scale)
    else:
        return input @ other
