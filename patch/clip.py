from typing import *

import transformers

from ._module import _checkpoint_module, _patch_module
from .functional import nn_layer_norm_forward, nn_linear_forward, quick_gelu_forward


def apply_patch_to_clip_model(
    model: "transformers.models.clip.modeling_clip.CLIPPreTrainedModel",
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
    from transformers.models.clip.modeling_clip import CLIPModel
    base_model: CLIPModel = model.base_model

    apply_patch_to_clip_text_model(
        base_model.text_model,
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

    apply_patch_to_clip_vision_model(
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


def apply_patch_to_clip_text_model(
    model: "transformers.models.clip.modeling_clip.CLIPPreTrainedModel",
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
    from transformers.models.clip.modeling_clip import CLIPTextModel, CLIPEncoderLayer
    base_model: CLIPTextModel = model.base_model

    for layer in base_model.text_model.encoder.layers:
        layer: CLIPEncoderLayer
        if norm:
            _patch_module(layer.layer_norm1, nn_layer_norm_forward, compress_kwargs=compress_kwargs)
            _patch_module(layer.layer_norm2, nn_layer_norm_forward, compress_kwargs=compress_kwargs)
        if attn_in:
            _patch_module(layer.self_attn.q_proj, nn_linear_forward, compress_kwargs=compress_kwargs)
            _patch_module(layer.self_attn.k_proj, nn_linear_forward, compress_kwargs=compress_kwargs)
            _patch_module(layer.self_attn.v_proj, nn_linear_forward, compress_kwargs=compress_kwargs)
        if attn_out:
            _patch_module(layer.self_attn.out_proj, nn_linear_forward, compress_kwargs=compress_kwargs)
        if mlp_in:
            _patch_module(layer.mlp.fc1, nn_linear_forward, compress_kwargs=compress_kwargs)
        if mlp_out:
            _patch_module(layer.mlp.fc2, nn_linear_forward, compress_kwargs=compress_kwargs)
        if act_fn:
            _patch_module(layer.mlp.activation_fn, quick_gelu_forward, compress_kwargs=compress_kwargs)
        if ckpt_attn:
            _checkpoint_module(layer.self_attn, compress_kwargs=compress_kwargs)
        if ckpt_mlp:
            _checkpoint_module(layer.mlp, compress_kwargs=compress_kwargs)
        if ckpt_layer:
            _checkpoint_module(layer, compress_kwargs=compress_kwargs)


def apply_patch_to_clip_vision_model(
    model: "transformers.models.clip.modeling_clip.CLIPPreTrainedModel",
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
    from transformers.models.clip.modeling_clip import CLIPVisionModel, CLIPEncoderLayer
    base_model: CLIPVisionModel = model.base_model

    for layer in base_model.vision_model.encoder.layers:
        layer: CLIPEncoderLayer
        if norm:
            _patch_module(layer.layer_norm1, nn_layer_norm_forward, compress_kwargs=compress_kwargs)
            _patch_module(layer.layer_norm2, nn_layer_norm_forward, compress_kwargs=compress_kwargs)
        if attn_in:
            _patch_module(layer.self_attn.q_proj, nn_linear_forward, compress_kwargs=compress_kwargs)
            _patch_module(layer.self_attn.k_proj, nn_linear_forward, compress_kwargs=compress_kwargs)
            _patch_module(layer.self_attn.v_proj, nn_linear_forward, compress_kwargs=compress_kwargs)
        if attn_out:
            _patch_module(layer.self_attn.out_proj, nn_linear_forward, compress_kwargs=compress_kwargs)
        if mlp_in:
            _patch_module(layer.mlp.fc1, nn_linear_forward, compress_kwargs=compress_kwargs)
        if mlp_out:
            _patch_module(layer.mlp.fc2, nn_linear_forward, compress_kwargs=compress_kwargs)
        if act_fn:
            _patch_module(layer.mlp.activation_fn, quick_gelu_forward, compress_kwargs=compress_kwargs)
        if ckpt_attn:
            _checkpoint_module(layer.self_attn, compress_kwargs=compress_kwargs)
        if ckpt_mlp:
            _checkpoint_module(layer.mlp, compress_kwargs=compress_kwargs)
        if ckpt_layer:
            _checkpoint_module(layer, compress_kwargs=compress_kwargs)
