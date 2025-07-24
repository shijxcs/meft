import warnings
from typing import *

import transformers

from ._module import _checkpoint_module, _patch_module
from .functional import nn_layer_norm_forward, nn_linear_forward, gelu_forward


def apply_patch_to_vit_model(
    model: "transformers.models.vit.modeling_vit.ViTPreTrainedModel",
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
    from transformers.models.vit.modeling_vit import ViTModel, ViTLayer
    base_model: ViTModel = model.base_model

    class CheckpointViTMLPWarning(UserWarning): ...
    warnings.simplefilter("once", CheckpointViTMLPWarning)

    for layer in base_model.encoder.layer:
        layer: ViTLayer
        if norm:
            _patch_module(layer.layernorm_before, nn_layer_norm_forward, compress_kwargs=compress_kwargs)
            _patch_module(layer.layernorm_after, nn_layer_norm_forward, compress_kwargs=compress_kwargs)
        if attn_in:
            _patch_module(layer.attention.attention.query, nn_linear_forward, compress_kwargs=compress_kwargs)
            _patch_module(layer.attention.attention.key, nn_linear_forward, compress_kwargs=compress_kwargs)
            _patch_module(layer.attention.attention.value, nn_linear_forward, compress_kwargs=compress_kwargs)
        if attn_out:
            _patch_module(layer.attention.output.dense, nn_linear_forward, compress_kwargs=compress_kwargs)
        if mlp_in:
            _patch_module(layer.intermediate.dense, nn_linear_forward, compress_kwargs=compress_kwargs)
        if mlp_out:
            _patch_module(layer.output.dense, nn_linear_forward, compress_kwargs=compress_kwargs)
        if act_fn:
            _patch_module(layer.intermediate.intermediate_act_fn, gelu_forward, compress_kwargs=compress_kwargs)
        if ckpt_attn:
            _checkpoint_module(layer.attention, compress_kwargs=compress_kwargs)
        if ckpt_mlp:
            warnings.warn("ViT only supports checkpointing the first layer of MLP.", CheckpointViTMLPWarning)
            _checkpoint_module(layer.intermediate, compress_kwargs=compress_kwargs)
        if ckpt_layer:
            _checkpoint_module(layer, compress_kwargs=compress_kwargs)
