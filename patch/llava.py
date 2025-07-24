from typing import *

import transformers

from .clip import apply_patch_to_clip_vision_model
from .llama import apply_patch_to_llama_model


def apply_patch_to_llava_model(
    model: "transformers.models.llava.modeling_llava.LlavaPreTrainedModel",
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
    from transformers.models.llava.modeling_llava import LlavaModel
    base_model: LlavaModel = model.base_model

    apply_patch_to_llama_model(
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

    apply_patch_to_clip_vision_model(
        base_model.vision_tower,
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
