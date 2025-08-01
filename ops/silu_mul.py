import torch
from torch import nn, Tensor

from ..compressed import CompressedTensor


class SiLUMulFunction(torch.autograd.Function):
    @torch.compile
    @staticmethod
    def forward(
        input: Tensor,
        alpha: Tensor,
        compress_kwargs: dict | None = None,
    ) -> Tensor:
        silu = nn.functional.silu(input)
        output = silu * alpha
        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, alpha, compress_kwargs = inputs
        if compress_kwargs is not None:
            ctx.save_for_backward(CompressedTensor(input, **compress_kwargs), CompressedTensor(alpha, **compress_kwargs))
        else:
            ctx.save_for_backward(input, alpha)

    @torch.compile
    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor | None, ...]:
        input, alpha = ctx.saved_tensors

        if isinstance(input, CompressedTensor):
            input = input.reconstruct()
        if isinstance(alpha, CompressedTensor):
            alpha = alpha.reconstruct()

        if ctx.needs_input_grad[0]:
            sigmoid = torch.sigmoid(input)
            grad_silu = grad_output * alpha
            grad_input = grad_silu * (sigmoid + sigmoid * (1 - sigmoid) * input)
        else:
            grad_input = None

        if ctx.needs_input_grad[1]:
            silu = nn.functional.silu(input)
            grad_alpha = grad_output * silu
        else:
            grad_alpha = None

        return grad_input, grad_alpha, None
