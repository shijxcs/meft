from typing import *

from torch import Tensor
import transformers

from ...ops.rms_norm import RMSNormFunction
from ...utils.dtype_utils import CastingMode


def gemma_rms_norm_forward(
    self: "transformers.models.gemma.modeling_gemma.GemmaRMSNorm",
    x: Tensor,
    compress_kwargs: Optional[Mapping[str, Any]] = None,
) -> Tensor:
    output, _ = RMSNormFunction.apply(
        x,
        None,
        self.weight,
        self.eps,
        CastingMode.ALL,
        compress_kwargs if self.training else None,
    )
    return output


def helium_rms_norm_forward(
    self: "transformers.models.helium.modeling_helium.HeliumRMSNorm",
    hidden_states: Tensor,
    compress_kwargs: Optional[Mapping[str, Any]] = None,
) -> Tensor:
    output, _ = RMSNormFunction.apply(
        hidden_states,
        None,
        self.weight,
        self.variance_epsilon,
        CastingMode.ALL,
        compress_kwargs if self.training else None,
    )
    return output


def llama4_text_rms_norm_forward(
    self: "transformers.models.llama4.modeling_llama4.Llama4TextRMSNorm",
    x: Tensor,
    compress_kwargs: Optional[Mapping[str, Any]] = None,
) -> Tensor:
    output, _ = RMSNormFunction.apply(
        x,
        None,
        self.weight,
        self.eps,
        CastingMode.INPUT,
        compress_kwargs if self.training else None,
    )
    return output


def nn_rms_norm_forward(
    self: "transformers.nn.RMSNorm",
    x: Tensor,
    compress_kwargs: Optional[Mapping[str, Any]] = None,
) -> Tensor:
    output, _ = RMSNormFunction.apply(
        x,
        self.normalized_shape,
        self.weight,
        self.eps,
        CastingMode.ALL,
        compress_kwargs if self.training else None,
    )
    return output


def t5_rms_norm_forward(
    self: "transformers.models.t5.modeling_t5.T5LayerNorm",
    hidden_states: Tensor,
    compress_kwargs: Optional[Mapping[str, Any]] = None,
) -> Tensor:
    output, _ = RMSNormFunction.apply(
        hidden_states,
        None,
        self.weight,
        self.variance_epsilon,
        CastingMode.INPUT,
        compress_kwargs if self.training else None,
    )
    return output
