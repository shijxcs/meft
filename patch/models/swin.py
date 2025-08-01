import warnings

import transformers

from ..functions import nn_layer_norm_forward, nn_linear_forward, gelu_forward
from ..patch import _checkpoint_module, _patch_module


def apply_patch_to_swin_model(
    model: "transformers.models.swin.modeling_swin.SwinPreTrainedModel",
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
    from transformers.models.swin.modeling_swin import SwinModel, SwinStage, SwinLayer
    base_model: SwinModel = model.base_model

    class CheckpointSwinMLPWarning(UserWarning): ...
    warnings.simplefilter("once", CheckpointSwinMLPWarning)

    for stage in base_model.encoder.layers:
        stage: SwinStage
        for layer in stage.blocks:
            layer: SwinLayer
            if norm:
                _patch_module(layer.layernorm_before, nn_layer_norm_forward, compress_kwargs=compress_kwargs)
                _patch_module(layer.layernorm_after, nn_layer_norm_forward, compress_kwargs=compress_kwargs)
            if attn_in:
                _patch_module(layer.attention.self.query, nn_linear_forward, compress_kwargs=compress_kwargs)
                _patch_module(layer.attention.self.key, nn_linear_forward, compress_kwargs=compress_kwargs)
                _patch_module(layer.attention.self.value, nn_linear_forward, compress_kwargs=compress_kwargs)
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
                warnings.warn("Swin only supports checkpointing the first layer of MLP.", CheckpointSwinMLPWarning)
                _checkpoint_module(layer.intermediate, compress_kwargs=compress_kwargs)
            if ckpt_layer:
                _checkpoint_module(layer, compress_kwargs=compress_kwargs)
