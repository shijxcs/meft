import warnings

import transformers

from ..functions import llama4_text_rms_norm_forward, nn_layer_norm_forward, nn_linear_forward, llama_mlp_forward, gelu_forward
from ..patch import _checkpoint_module, _patch_module


def apply_patch_to_llama4_model(
    model: "transformers.models.llama4.modeling_llama4.Llama4PreTrainedModel",
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
    from transformers.models.llama4.modeling_llama4 import Llama4ForConditionalGeneration
    base_model: Llama4ForConditionalGeneration = model.base_model

    apply_patch_to_llama4_text_model(
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

    apply_patch_to_llama4_vision_model(
        base_model.vision_model,
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


def apply_patch_to_llama4_text_model(
    model: "transformers.models.llama4.modeling_llama4.Llama4PreTrainedModel",
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
    from transformers.models.llama4.modeling_llama4 import Llama4TextModel, Llama4TextDecoderLayer, Llama4TextMoe, Llama4TextMLP
    base_model: Llama4TextModel = model.base_model

    class PatchLlama4TextExpertsWarning(UserWarning): ...
    warnings.simplefilter("once", PatchLlama4TextExpertsWarning)

    def apply_patch_to_llama4_text_mlp_module(
        module: Llama4TextMLP,
        mlp_in: bool = False,
        mlp_out: bool = False,
        act_fn: bool = False,
        compress_kwargs: dict | None = None,
    ) -> None:
        if mlp_in:
            _patch_module(module.gate_proj, nn_linear_forward, compress_kwargs=compress_kwargs)
            _patch_module(module.up_proj, nn_linear_forward, compress_kwargs=compress_kwargs)
        if mlp_out:
            _patch_module(module.down_proj, nn_linear_forward, compress_kwargs=compress_kwargs)
        if act_fn:
            _patch_module(module, llama_mlp_forward, compress_kwargs=compress_kwargs)

    for layer in base_model.layers:
        layer: Llama4TextDecoderLayer
        if norm:
            _patch_module(layer.input_layernorm, llama4_text_rms_norm_forward, compress_kwargs=compress_kwargs)
            _patch_module(layer.post_attention_layernorm, llama4_text_rms_norm_forward, compress_kwargs=compress_kwargs)
        if attn_in:
            _patch_module(layer.self_attn.q_proj, nn_linear_forward, compress_kwargs=compress_kwargs)
            _patch_module(layer.self_attn.k_proj, nn_linear_forward, compress_kwargs=compress_kwargs)
            _patch_module(layer.self_attn.v_proj, nn_linear_forward, compress_kwargs=compress_kwargs)
        if attn_out:
            _patch_module(layer.self_attn.o_proj, nn_linear_forward, compress_kwargs=compress_kwargs)
        if isinstance(layer.feed_forward, Llama4TextMoe):
            if mlp_in or mlp_out or act_fn:
                warnings.warn("No patch supported for Llama4TextExperts.", PatchLlama4TextExpertsWarning)
            apply_patch_to_llama4_text_mlp_module(layer.feed_forward.shared_expert, mlp_in=mlp_in, mlp_out=mlp_out, act_fn=act_fn, compress_kwargs=compress_kwargs)
        else:
            apply_patch_to_llama4_text_mlp_module(layer.feed_forward, mlp_in=mlp_in, mlp_out=mlp_out, act_fn=act_fn, compress_kwargs=compress_kwargs)
        if ckpt_attn:
            _checkpoint_module(layer.self_attn, compress_kwargs=compress_kwargs)
        if ckpt_mlp:
            _checkpoint_module(layer.feed_forward, compress_kwargs=compress_kwargs)
        if ckpt_layer:
            _checkpoint_module(layer, compress_kwargs=compress_kwargs)


def apply_patch_to_llama4_vision_model(
    model: "transformers.models.llama4.modeling_llama4.Llama4PreTrainedModel",
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
    from transformers.models.llama4.modeling_llama4 import Llama4VisionModel, Llama4VisionEncoderLayer
    base_model: Llama4VisionModel = model.base_model

    for layer in base_model.model.layers:
        layer: Llama4VisionEncoderLayer
        if norm:
            _patch_module(layer.input_layernorm, nn_layer_norm_forward, compress_kwargs=compress_kwargs)
            _patch_module(layer.post_attention_layernorm, nn_layer_norm_forward, compress_kwargs=compress_kwargs)
        if attn_in:
            _patch_module(layer.self_attn.q_proj, nn_linear_forward, compress_kwargs=compress_kwargs)
            _patch_module(layer.self_attn.k_proj, nn_linear_forward, compress_kwargs=compress_kwargs)
            _patch_module(layer.self_attn.v_proj, nn_linear_forward, compress_kwargs=compress_kwargs)
        if attn_out:
            _patch_module(layer.self_attn.o_proj, nn_linear_forward, compress_kwargs=compress_kwargs)
        if mlp_in:
            _patch_module(layer.mlp.fc1, nn_linear_forward, compress_kwargs=compress_kwargs)
        if mlp_out:
            _patch_module(layer.mlp.fc2, nn_linear_forward, compress_kwargs=compress_kwargs)
        if act_fn:
            _patch_module(layer.mlp.activation_fn, gelu_forward, compress_kwargs=compress_kwargs)
        if ckpt_attn:
            _checkpoint_module(layer.self_attn, compress_kwargs=compress_kwargs)
        if ckpt_mlp:
            _checkpoint_module(layer.mlp, compress_kwargs=compress_kwargs)
        if ckpt_layer:
            _checkpoint_module(layer, compress_kwargs=compress_kwargs)
