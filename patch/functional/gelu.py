from typing import *

from torch import Tensor
import transformers.activations

from ...ops.gelu import GELUFunction


def gelu_forward(
    self: "transformers.activations.GELUActivation",
    input: Tensor,
    compress_kwargs: Optional[Mapping[str, Any]] = None,
) -> Tensor:
    return GELUFunction.apply(
        input,
        "none",
        compress_kwargs if self.training else None,
    )


def gelu_new_forward(
    self: "transformers.activations.NewGELUActivation",
    input: Tensor,
    compress_kwargs: Optional[Mapping[str, Any]] = None,
) -> Tensor:
    return GELUFunction.apply(
        input,
        "tanh",
        compress_kwargs if self.training else None,
    )


def gelu_pytorch_tanh_forward(
    self: "transformers.activations.PytorchGELUTanh",
    input: Tensor,
    compress_kwargs: Optional[Mapping[str, Any]] = None,
) -> Tensor:
    return GELUFunction.apply(
        input,
        "tanh",
        compress_kwargs if self.training else None,
    )


def quick_gelu_forward(
    self: "transformers.activations.QuickGELUActivation",
    input: Tensor,
    compress_kwargs: Optional[Mapping[str, Any]] = None,
) -> Tensor:
    return GELUFunction.apply(
        input,
        "sigmoid",
        compress_kwargs if self.training else None,
    )
