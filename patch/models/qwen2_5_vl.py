import transformers

from ..functions import t5_rms_norm_forward, nn_layer_norm_forward, nn_linear_forward, llama_mlp_forward
from ..patch import _checkpoint_module, _patch_module


def apply_patch_to_qwen2_5_vl_model(
    model: "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLPreTrainedModel",
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
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel
    base_model: Qwen2_5_VLModel = model.base_model

    apply_patch_to_qwen2_5_vl_text_model(
        base_model.language_model,
        norm=norm,
        attn_in=attn_in,
        attn_out=attn_out,
        mlp_in=mlp_in,
        mlp_out=mlp_out,
        act_fn=act_fn,
        ckpt_attn=ckpt_attn,
        ckpt_mlp=ckpt_mlp,
        ckpt_layer=ckpt_layer,
        compress_kwargs=compress_kwargs,
    )

    apply_patch_to_qwen2_5_vl_vision_model(
        base_model.visual,
        norm=norm,
        attn_in=attn_in,
        attn_out=attn_out,
        mlp_in=mlp_in,
        mlp_out=mlp_out,
        act_fn=act_fn,
        ckpt_attn=ckpt_attn,
        ckpt_mlp=ckpt_mlp,
        ckpt_layer=ckpt_layer,
        compress_kwargs=compress_kwargs,
    )


def apply_patch_to_qwen2_5_vl_text_model(
    model: "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLPreTrainedModel",
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
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLTextModel, Qwen2_5_VLDecoderLayer, Qwen2_5_VLAttention
    base_model: Qwen2_5_VLTextModel = model.base_model

    for layer in base_model.layers:
        layer: Qwen2_5_VLDecoderLayer
        self_attn: Qwen2_5_VLAttention = layer.self_attn
        if norm:
            _patch_module(layer.input_layernorm, t5_rms_norm_forward, compress_kwargs=compress_kwargs)
            _patch_module(layer.post_attention_layernorm, t5_rms_norm_forward, compress_kwargs=compress_kwargs)
        if attn_in:
            _patch_module(self_attn.q_proj, nn_linear_forward, compress_kwargs=compress_kwargs)
            _patch_module(self_attn.k_proj, nn_linear_forward, compress_kwargs=compress_kwargs)
            _patch_module(self_attn.v_proj, nn_linear_forward, compress_kwargs=compress_kwargs)
        if attn_out:
            _patch_module(self_attn.o_proj, nn_linear_forward, compress_kwargs=compress_kwargs)
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


def apply_patch_to_qwen2_5_vl_vision_model(
    model: "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLPreTrainedModel",
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
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel, Qwen2_5_VLVisionBlock, Qwen2_5_VLVisionAttention
    base_model: Qwen2_5_VisionTransformerPretrainedModel = model.base_model

    for layer in base_model.blocks:
        layer: Qwen2_5_VLVisionBlock
        attn: Qwen2_5_VLVisionAttention = layer.attn
        if norm:
            _patch_module(layer.norm1, nn_layer_norm_forward, compress_kwargs=compress_kwargs)
            _patch_module(layer.norm2, nn_layer_norm_forward, compress_kwargs=compress_kwargs)
        if attn_in:
            _patch_module(attn.qkv, nn_linear_forward, compress_kwargs=compress_kwargs)
        if attn_out:
            _patch_module(attn.proj, nn_linear_forward, compress_kwargs=compress_kwargs)
        if mlp_in:
            _patch_module(layer.mlp.gate_proj, nn_linear_forward, compress_kwargs=compress_kwargs)
            _patch_module(layer.mlp.up_proj, nn_linear_forward, compress_kwargs=compress_kwargs)
        if mlp_out:
            _patch_module(layer.mlp.down_proj, nn_linear_forward, compress_kwargs=compress_kwargs)
        if act_fn:
            _patch_module(layer.mlp, llama_mlp_forward, compress_kwargs=compress_kwargs)
        if ckpt_attn:
            _checkpoint_module(layer.attn, compress_kwargs=compress_kwargs)
        if ckpt_mlp:
            _checkpoint_module(layer.mlp, compress_kwargs=compress_kwargs)
        if ckpt_layer:
            _checkpoint_module(layer, compress_kwargs=compress_kwargs)
