from torch import Tensor
import transformers

from ...ops.silu_mul import SiLUMulFunction


def glm_mlp_forward(
    self: "transformers.models.glm.modeling_glm.GlmMLP",
    hidden_states: Tensor,
    compress_kwargs: dict | None = None,
) -> Tensor:
    up_states = self.gate_up_proj(hidden_states)
    gate, up_states = up_states.chunk(2, dim=-1)
    up_states = SiLUMulFunction.apply(
        gate,
        up_states,
        compress_kwargs if self.training else None,
    )
    return self.down_proj(up_states)


def llama_mlp_forward(
    self: "transformers.models.llama.modeling_llama.LlamaMLP",
    x: Tensor,
    compress_kwargs: dict | None = None,
) -> Tensor:
    intermediate = SiLUMulFunction.apply(
        self.gate_proj(x),
        self.up_proj(x),
        compress_kwargs if self.training else None,
    )
    return self.down_proj(intermediate)


def mixtral_mlp_forward(
    self: "transformers.models.mixtral.modeling_mixtral.MixtralBlockSparseTop2MLP",
    x: Tensor,
    compress_kwargs: dict | None = None,
) -> Tensor:
    intermediate = SiLUMulFunction.apply(
        self.w1(x),
        self.w3(x),
        compress_kwargs if self.training else None,
    )
    return self.w2(intermediate)
