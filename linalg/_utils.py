from math import sqrt

from torch import Tensor


def scaled_matmul(input: Tensor, other: Tensor):
    scale = sqrt(1 / input.shape[-1])
    return (input * scale) @ (other * scale)
