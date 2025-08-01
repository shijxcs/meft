from torch import nn, Tensor
import transformers

from ...ops.rms_norm import RMSNormFunction
from ...ops.utils import CastingMode


def gemma_rms_norm_forward(
    self: "transformers.models.gemma.modeling_gemma.GemmaRMSNorm",
    x: Tensor,
    compress_kwargs: dict | None = None,
) -> Tensor:
    output = RMSNormFunction.apply(
        x,
        None,
        self.weight,
        self.eps,
        CastingMode.ALL,
        compress_kwargs if self.training else None,
    )
    output, _ = output
    return output


def helium_rms_norm_forward(
    self: "transformers.models.helium.modeling_helium.HeliumRMSNorm",
    hidden_states: Tensor,
    compress_kwargs: dict | None = None,
) -> Tensor:
    output = RMSNormFunction.apply(
        hidden_states,
        None,
        self.weight,
        self.variance_epsilon,
        CastingMode.ALL,
        compress_kwargs if self.training else None,
    )
    output, _ = output
    return output


def llama4_text_rms_norm_forward(
    self: "transformers.models.llama4.modeling_llama4.Llama4TextRMSNorm",
    x: Tensor,
    compress_kwargs: dict | None = None,
) -> Tensor:
    output = RMSNormFunction.apply(
        x,
        None,
        self.weight,
        self.eps,
        CastingMode.INPUT,
        compress_kwargs if self.training else None,
    )
    output, _ = output
    return output


def nn_rms_norm_forward(
    self: "nn.RMSNorm",
    x: Tensor,
    compress_kwargs: dict | None = None,
) -> Tensor:
    output = RMSNormFunction.apply(
        x,
        self.normalized_shape,
        self.weight,
        self.eps,
        CastingMode.ALL,
        compress_kwargs if self.training else None,
    )
    output, _ = output
    return output


def t5_rms_norm_forward(
    self: "transformers.models.t5.modeling_t5.T5LayerNorm",
    hidden_states: Tensor,
    compress_kwargs: dict | None = None,
) -> Tensor:
    output = RMSNormFunction.apply(
        hidden_states,
        None,
        self.weight,
        self.variance_epsilon,
        CastingMode.INPUT,
        compress_kwargs if self.training else None,
    )
    output, _ = output
    return output
