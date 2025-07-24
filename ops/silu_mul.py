from typing import *

import torch
from torch import nn, Tensor

from ..compressed import CompressedTensor


class SiLUMulFunction(torch.autograd.Function):
    @torch.compile
    @staticmethod
    def forward(
        input: Tensor,
        alpha: Tensor,
        compress_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> Tensor:
        silu = nn.functional.silu(input)
        output = silu * alpha
        return output

    @torch.compile
    @staticmethod
    def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
        input, alpha, compress_kwargs = inputs
        if compress_kwargs is not None:
            ctx.save_for_backward(CompressedTensor(input, **compress_kwargs), CompressedTensor(alpha, **compress_kwargs))
        else:
            ctx.save_for_backward(input, alpha)

    @torch.compile
    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Tensor, Tensor, None]:
        input, alpha = ctx.saved_tensors

        if isinstance(input, CompressedTensor):
            input = input.reconstruct()
        if isinstance(alpha, CompressedTensor):
            alpha = alpha.reconstruct()

        silu = nn.functional.silu(input)
        sigmoid = torch.sigmoid(input)

        grad_silu = grad_output * alpha
        grad_alpha = grad_output * silu
        grad_input = grad_silu * (sigmoid + sigmoid * (1 - sigmoid) * input)
        return grad_input, grad_alpha, None
