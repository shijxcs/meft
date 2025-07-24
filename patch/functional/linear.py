from typing import *

from torch import nn, Tensor

from ...ops.linear import LinearFunction


def nn_linear_forward(
    self: "nn.Linear",
    input: Tensor,
    compress_kwargs: Optional[Mapping[str, Any]] = None,
) -> Tensor:
    return LinearFunction.apply(
        input,
        self.weight,
        self.bias,
        compress_kwargs if self.training else None,
    )
