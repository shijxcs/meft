from torch import nn, Tensor
import transformers

from ...ops.layer_norm import LayerNormFunction
from ...ops.utils import CastingMode


def cohere_layer_norm_forward(
    self: "transformers.models.cohere.modeling_cohere.CohereLayerNorm",
    hidden_states: Tensor,
    compress_kwargs: dict | None = None,
) -> Tensor:
    output = LayerNormFunction.apply(
        hidden_states,
        None,
        self.weight,
        None,
        self.variance_epsilon,
        CastingMode.ALL,
        compress_kwargs if self.training else None,
    )
    output, _ = output
    return output


def nn_layer_norm_forward(
    self: "nn.LayerNorm",
    input: Tensor,
    compress_kwargs: dict | None = None,
) -> Tensor:
    output = LayerNormFunction.apply(
        input,
        self.normalized_shape,
        self.weight,
        self.bias,
        self.eps,
        CastingMode.ALL,
        compress_kwargs if self.training else None,
    )
    output, _ = output
    return output


def olmo_layer_norm_forward(
    self: "transformers.models.olmo.modeling_olmo.OlmoLayerNorm",
    hidden_states: Tensor,
    compress_kwargs: dict | None = None,
) -> Tensor:
    output = LayerNormFunction.apply(
        hidden_states,
        self.normalized_shape,
        None,
        None,
        1e-5,
        CastingMode.ALL,
        compress_kwargs if self.training else None,
    )
    output, _ = output
    return output
