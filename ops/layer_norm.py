from typing import *

import torch
from torch import Tensor

from ..compressed import CompressedTensor
from ..utils.dtype_utils import CastingMode, get_float_eps, convert_dtype, promote_dtype


class LayerNormFunction(torch.autograd.Function):
    @torch.compile
    @staticmethod
    def forward(
        input: Tensor,
        normalized_shape: list[int],
        weight: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        eps: Optional[float] = None,
        casting_mode: CastingMode = CastingMode.NONE,
        compress_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[Tensor, Tensor]:
        if normalized_shape is not None:
            assert input.shape[-len(normalized_shape):] == tuple(normalized_shape)
            reduction_dim = list(range(-len(normalized_shape), 0))
        else:
            reduction_dim = -1

        input_dtype = input.dtype

        if casting_mode == CastingMode.INPUT:
            input, = promote_dtype(input, dtype=torch.float32)

        if casting_mode == CastingMode.ALL:
            input, weight, bias = promote_dtype(input, weight, bias, dtype=torch.float32)

        if eps is None:
            eps = get_float_eps(input.dtype)

        mean = input.mean(dim=reduction_dim, keepdim=True)
        var = input.var(dim=reduction_dim, keepdim=True, unbiased=False)
        rstd = torch.rsqrt(var + eps)
        output = (input - mean) * rstd

        if casting_mode == CastingMode.INPUT:
            output, = convert_dtype(output, dtype=input_dtype)

        if weight is not None:
            output = output * weight

        if bias is not None:
            output = output + bias

        if casting_mode == CastingMode.ALL:
            output, = convert_dtype(output, dtype=input_dtype)
        return output, rstd

    @torch.compile
    @staticmethod
    def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
        input, normalized_shape, weight, bias, eps, casting_mode, compress_kwargs = inputs
        output, rstd = output
        ctx.normalized_shape = normalized_shape
        ctx.casting_mode = casting_mode
        if compress_kwargs is not None:
            ctx.save_for_backward(CompressedTensor(output, **compress_kwargs), weight, bias, rstd)
        else:
            ctx.save_for_backward(output, weight, bias, rstd)

    @torch.compile
    @staticmethod
    def backward(ctx, grad_output: Tensor, _0: Any) -> Tuple[Tensor, None, Optional[Tensor], Optional[Tensor], None, None]:
        if ctx.normalized_shape is not None:
            reduction_dim = list(range(-len(ctx.normalized_shape), 0))
        else:
            reduction_dim = -1
        output, weight, bias, rstd, = ctx.saved_tensors

        if isinstance(output, CompressedTensor):
            output = output.reconstruct()

        input_dtype = output.dtype

        if ctx.casting_mode == CastingMode.ALL:
            grad_output, output = promote_dtype(grad_output, output, dtype=torch.float32)

        if bias is not None:
            input_normalized = output - bias
            grad_bias = grad_output
        else:
            input_normalized = output
            grad_bias = None

        if weight is not None:
            input_normalized = input_normalized / weight
            grad_weight = (grad_output * input_normalized).view(-1, *weight.shape).sum(dim=0)
            grad_input_normalized = grad_output * weight
        else:
            grad_weight = None
            grad_input_normalized = grad_output

        if ctx.casting_mode == CastingMode.INPUT:
            grad_input_normalized, input_normalized = promote_dtype(grad_input_normalized, input_normalized, dtype=torch.float32)

        grad_input = rstd * (
            grad_input_normalized
            - grad_input_normalized.mean(dim=reduction_dim, keepdim=True)
            - input_normalized * (grad_input_normalized * input_normalized).mean(dim=reduction_dim, keepdim=True)
        )

        grad_input, = convert_dtype(grad_input, dtype=input_dtype)

        return grad_input, None, grad_weight, grad_bias, None, None, None
