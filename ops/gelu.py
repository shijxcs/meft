import math

import torch
from torch import Tensor

from ..compressed import CompressedTensor


class GELUFunction(torch.autograd.Function):
    @torch.compile
    @staticmethod
    def forward(
        input: Tensor,
        approximate: str = "none",
        compress_kwargs: dict | None = None,
    ) -> Tensor:
        if approximate == "none":
            return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))
        elif approximate == "tanh":
            return input * 0.5 * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3))))
        elif approximate == "sigmoid":
            return input * torch.sigmoid(1.702 * input)
        else:
            raise ValueError("Unexpected value of argument `approximate`, must be `'none'`, `'tanh'` or `'sigmoid'`.")

    @staticmethod
    def setup_context(ctx, inputs: tuple, output: Tensor) -> None:
        input, approximate, compress_kwargs = inputs
        ctx.approximate = approximate
        if compress_kwargs is not None:
            ctx.save_for_backward(CompressedTensor(input, **compress_kwargs))
        else:
            ctx.save_for_backward(input)

    @torch.compile
    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor | None, ...]:
        input, = ctx.saved_tensors

        if isinstance(input, CompressedTensor):
            input = input.reconstruct()

        if ctx.needs_input_grad[0]:
            if ctx.approximate == "none":
                cdf = 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))
                deriv_cdf = (1.0 / math.sqrt(2.0 * math.pi)) * torch.exp(-0.5 * torch.pow(input, 2))
            elif ctx.approximate == "tanh":
                cdf = 0.5 * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3))))
                deriv_cdf = 2.0 * cdf * (1.0 - cdf) * math.sqrt(2.0 / math.pi) * (1.0 + 0.134145 * torch.pow(input, 2))
            elif ctx.approximate == "sigmoid":
                cdf = torch.sigmoid(1.702 * input)
                deriv_cdf = 1.702 * cdf * (1.0 - cdf)
            else:
                raise ValueError("Unexpected value of argument `approximate`, must be `'none'`, `'tanh'` or `'sigmoid'`.")
            grad_input = grad_output * (cdf + input * deriv_cdf)
        else:
            grad_input = None

        return grad_input, None, None
