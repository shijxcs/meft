from typing import *

import torch
from torch import nn, Tensor

from ..compressed import CompressedTensor


class LinearFunction(torch.autograd.Function):
    @torch.compile
    @staticmethod
    def forward(
        input: Tensor,
        weight: Tensor,
        bias: Optional[Tensor] = None,
        compress_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> Tensor:
        return nn.functional.linear(input, weight, bias)

    @torch.compile
    @staticmethod
    def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
        input, weight, bias, compress_kwargs = inputs
        
        ctx.device_type = input.device.type
        ctx.autocast_kwargs = {
            "dtype": torch.get_autocast_dtype(ctx.device_type),
            "enabled": torch.is_autocast_enabled(ctx.device_type),
            "cache_enabled": torch.is_autocast_cache_enabled(),
        }
        if compress_kwargs is not None:
            ctx.save_for_backward(CompressedTensor(input, **compress_kwargs), weight)
        else:
            ctx.save_for_backward(input, weight)
        ctx.bias = (bias is not None)

    @torch.compile
    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Any]:
        input, weight = ctx.saved_tensors

        if isinstance(input, CompressedTensor):
            input = input.reconstruct()

        with torch.autocast(ctx.device_type, **ctx.autocast_kwargs):
            grad_input = grad_output @ weight
            grad_weight = grad_output.view(-1, grad_output.shape[-1]).t() @ input.view(-1, input.shape[-1])
            grad_bias = grad_output if ctx.bias else None

        return grad_input, grad_weight, grad_bias, None
