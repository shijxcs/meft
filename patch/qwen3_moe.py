from typing import *

import transformers

from ._module import _checkpoint_module, _patch_module
from .functional import t5_rms_norm_forward, nn_linear_forward, llama_mlp_forward


def apply_patch_to_qwen3_moe_model(
    model: "transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoePreTrainedModel",
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
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeModel, Qwen3MoeDecoderLayer, Qwen3MoeSparseMoeBlock, Qwen3MoeMLP
    base_model: Qwen3MoeModel = model.base_model

    def apply_patch_to_qwen3_moe_mlp_module(
        module: Qwen3MoeMLP,
        mlp_in: bool = False,
        mlp_out: bool = False,
        act_fn: bool = False,
        compress_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if mlp_in:
            _patch_module(module.gate_proj, nn_linear_forward, compress_kwargs=compress_kwargs)
            _patch_module(module.up_proj, nn_linear_forward, compress_kwargs=compress_kwargs)
        if mlp_out:
            _patch_module(module.down_proj, nn_linear_forward, compress_kwargs=compress_kwargs)
        if act_fn:
            _patch_module(module, llama_mlp_forward, compress_kwargs=compress_kwargs)

    for layer in base_model.layers:
        layer: Qwen3MoeDecoderLayer
        if norm:
            _patch_module(layer.input_layernorm, t5_rms_norm_forward, compress_kwargs=compress_kwargs)
            _patch_module(layer.post_attention_layernorm, t5_rms_norm_forward, compress_kwargs=compress_kwargs)
        if attn_in:
            _patch_module(layer.self_attn.q_proj, nn_linear_forward, compress_kwargs=compress_kwargs)
            _patch_module(layer.self_attn.k_proj, nn_linear_forward, compress_kwargs=compress_kwargs)
            _patch_module(layer.self_attn.v_proj, nn_linear_forward, compress_kwargs=compress_kwargs)
        if attn_out:
            _patch_module(layer.self_attn.o_proj, nn_linear_forward, compress_kwargs=compress_kwargs)
        if isinstance(layer.mlp.experts, Qwen3MoeSparseMoeBlock):
            for expert in layer.mlp:
                expert: Qwen3MoeMLP
                apply_patch_to_qwen3_moe_mlp_module(expert, mlp_in=mlp_in, mlp_out=mlp_out, act_fn=act_fn, compress_kwargs=compress_kwargs)
            if mlp_in:
                _patch_module(layer.mlp.gate, nn_linear_forward, compress_kwargs=compress_kwargs)
        else:
            apply_patch_to_qwen3_moe_mlp_module(layer.mlp, mlp_in=mlp_in, mlp_out=mlp_out, act_fn=act_fn, compress_kwargs=compress_kwargs)
        if ckpt_attn:
            _checkpoint_module(layer.self_attn, compress_kwargs=compress_kwargs)
        if ckpt_mlp:
            _checkpoint_module(layer.mlp, compress_kwargs=compress_kwargs)
        if ckpt_layer:
            _checkpoint_module(layer, compress_kwargs=compress_kwargs)
