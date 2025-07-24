from typing import *

import transformers

from ._module import _checkpoint_module, _patch_module
from .functional import t5_rms_norm_forward, nn_linear_forward, llama_mlp_forward


def apply_patch_to_mistral_model(
    model: "transformers.models.mistral.modeling_mistral.MistralPreTrainedModel",
    norm: bool = False,
    attn_in: bool = False,
    attn_out: bool = False,
    mlp_in: bool = False,
    mlp_out: bool = False,
    act_fn: bool = False,
    ckpt_attn: bool = False,
    ckpt_mlp: bool = False,
    ckpt_layer: bool = False,
    compress_kwargs: Optional[Mapping[str, Any]] = None,
) -> None:
    from transformers.models.mistral.modeling_mistral import MistralModel, MistralDecoderLayer
    base_model: MistralModel = model.base_model

    for layer in base_model.layers:
        layer: MistralDecoderLayer
        if norm:
            _patch_module(layer.input_layernorm, t5_rms_norm_forward, compress_kwargs=compress_kwargs)
            _patch_module(layer.post_attention_layernorm, t5_rms_norm_forward, compress_kwargs=compress_kwargs)
        if attn_in:
            _patch_module(layer.self_attn.q_proj, nn_linear_forward, compress_kwargs=compress_kwargs)
            _patch_module(layer.self_attn.k_proj, nn_linear_forward, compress_kwargs=compress_kwargs)
            _patch_module(layer.self_attn.v_proj, nn_linear_forward, compress_kwargs=compress_kwargs)
        if attn_out:
            _patch_module(layer.self_attn.o_proj, nn_linear_forward, compress_kwargs=compress_kwargs)
        if mlp_in:
            _patch_module(layer.mlp.gate_proj, nn_linear_forward, compress_kwargs=compress_kwargs)
            _patch_module(layer.mlp.up_proj, nn_linear_forward, compress_kwargs=compress_kwargs)
        if mlp_out:
            _patch_module(layer.mlp.down_proj, nn_linear_forward, compress_kwargs=compress_kwargs)
        if act_fn:
            _patch_module(layer.mlp, llama_mlp_forward, compress_kwargs=compress_kwargs)
        if ckpt_attn:
            _checkpoint_module(layer.self_attn, compress_kwargs=compress_kwargs)
        if ckpt_mlp:
            _checkpoint_module(layer.mlp, compress_kwargs=compress_kwargs)
        if ckpt_layer:
            _checkpoint_module(layer, compress_kwargs=compress_kwargs)
