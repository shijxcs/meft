from torch import nn, Tensor

from ...ops.linear import LinearFunction


def nn_linear_forward(
    self: "nn.Linear",
    input: Tensor,
    compress_kwargs: dict | None = None,
) -> Tensor:
    output = LinearFunction.apply(
        input,
        self.weight,
        self.bias,
        compress_kwargs if self.training else None,
    )
    return output
