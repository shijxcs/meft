import transformers

from ..functions import t5_rms_norm_forward, nn_linear_forward, glm_mlp_forward
from ..patch import _checkpoint_module, _patch_module


def apply_patch_to_glm4_model(
    model: "transformers.models.glm4.modeling_glm4.Glm4PreTrainedModel",
    norm: bool = False,
    attn_in: bool = False,
    attn_out: bool = False,
    mlp_in: bool = False,
    mlp_out: bool = False,
    act_fn: bool = False,
    ckpt_attn: bool = False,
    ckpt_mlp: bool = False,
    ckpt_layer: bool = False,
    compress_kwargs: dict | None = None,
) -> None:
    from transformers.models.glm4.modeling_glm4 import Glm4Model, Glm4DecoderLayer
    base_model: Glm4Model = model.base_model

    for layer in base_model.layers:
        layer: Glm4DecoderLayer
        if norm:
            _patch_module(layer.input_layernorm, t5_rms_norm_forward, compress_kwargs=compress_kwargs)
            _patch_module(layer.post_self_attn_layernorm, t5_rms_norm_forward, compress_kwargs=compress_kwargs)
            _patch_module(layer.post_attention_layernorm, t5_rms_norm_forward, compress_kwargs=compress_kwargs)
            _patch_module(layer.post_mlp_layernorm, t5_rms_norm_forward, compress_kwargs=compress_kwargs)
        if attn_in:
            _patch_module(layer.self_attn.q_proj, nn_linear_forward, compress_kwargs=compress_kwargs)
            _patch_module(layer.self_attn.k_proj, nn_linear_forward, compress_kwargs=compress_kwargs)
            _patch_module(layer.self_attn.v_proj, nn_linear_forward, compress_kwargs=compress_kwargs)
        if attn_out:
            _patch_module(layer.self_attn.o_proj, nn_linear_forward, compress_kwargs=compress_kwargs)
        if mlp_in:
            _patch_module(layer.mlp.gate_up_proj, nn_linear_forward, compress_kwargs=compress_kwargs)
        if mlp_out:
            _patch_module(layer.mlp.down_proj, nn_linear_forward, compress_kwargs=compress_kwargs)
        if act_fn:
            _patch_module(layer.mlp, glm_mlp_forward, compress_kwargs=compress_kwargs)
        if ckpt_attn:
            _checkpoint_module(layer.self_attn, compress_kwargs=compress_kwargs)
        if ckpt_mlp:
            _checkpoint_module(layer.mlp, compress_kwargs=compress_kwargs)
        if ckpt_layer:
            _checkpoint_module(layer, compress_kwargs=compress_kwargs)
