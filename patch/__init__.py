import warnings
from collections.abc import Iterable

from transformers.modeling_utils import PreTrainedModel

from .models.cohere import *
from .models.cohere2 import *
from .models.deepseek_v3 import *
from .models.gemma import *
from .models.gemma2 import *
from .models.gemma3 import *
from .models.glm4 import *
from .models.granite import *
from .models.llama import *
from .models.llama4 import *
from .models.llava import *
from .models.mistral import *
from .models.mixtral import *
from .models.mllama import *
from .models.olmo import *
from .models.olmo2 import *
from .models.paligemma import *
from .models.phi import *
from .models.phi3 import *
from .models.qwen2 import *
from .models.qwen2_vl import *
from .models.qwen2_5_vl import *
from .models.qwen3 import *
from .models.qwen3_moe import *
from .models.siglip import *
from .models.siglip2 import *
from .models.swin import *
from .models.vit import *


MODEL_TYPE_TO_APPLY_FN = {
    "cohere": apply_patch_to_cohere_model,
    "cohere2": apply_patch_to_cohere2_model,
    "deepseek_v3": apply_patch_to_deepseek_v3_model,
    "gemma": apply_patch_to_gemma_model,
    "gemma2": apply_patch_to_gemma2_model,
    "gemma3": apply_patch_to_gemma3_model,
    "gemma3_text": apply_patch_to_gemma3_text_model,
    "granite": apply_patch_to_granite_model,
    "glm4": apply_patch_to_glm4_model,
    "llama": apply_patch_to_llama_model,
    "llama4": apply_patch_to_llama4_model,
    "llama4_text": apply_patch_to_llama4_text_model,
    "llama4_vision_model": apply_patch_to_llama4_vision_model,
    "llava": apply_patch_to_llava_model,
    "mistral": apply_patch_to_mistral_model,
    "mixtral": apply_patch_to_mixtral_model,
    "mllama": apply_patch_to_mllama_model,
    "mllama_text_model": apply_patch_to_mllama_text_model,
    "olmo": apply_patch_to_olmo_model,
    "olmo2": apply_patch_to_olmo2_model,
    "paligemma": apply_patch_to_paligemma_model,
    "phi": apply_patch_to_phi_model,
    "phi3": apply_patch_to_phi3_model,
    "qwen2": apply_patch_to_qwen2_model,
    "qwen2_vl": apply_patch_to_qwen2_vl_model,
    "qwen2_vl_text": apply_patch_to_qwen2_vl_text_model,
    "qwen2_5_vl": apply_patch_to_qwen2_5_vl_model,
    "qwen2_5_vl_text": apply_patch_to_qwen2_5_vl_model,
    "qwen3": apply_patch_to_qwen3_model,
    "qwen3_moe": apply_patch_to_qwen3_moe_model,
    "siglip": apply_patch_to_siglip_model,
    "siglip_text_model": apply_patch_to_siglip_text_model,
    "siglip_vision_model": apply_patch_to_siglip_vision_model,
    "siglip2": apply_patch_to_siglip2_model,
    "siglip2_text_model": apply_patch_to_siglip2_text_model,
    "siglip2_vision_model": apply_patch_to_siglip2_vision_model,
    "swin": apply_patch_to_swin_model,
    "vit": apply_patch_to_vit_model,
}


def apply_patch_to_model(
    model: PreTrainedModel,
    patch_locations: Iterable | None = None,
    compress_kwargs: dict | None = None,
) -> None:
    """
    Apply patch to modules by replacing their forward methods. Note that
    replacing forward methods should be after replacing the module classes
    (e.g. after applying liger kenel).
    """
    model_type = getattr(model, "config", None) and getattr(model.config, "model_type", None)

    if not model_type:
        warnings.warn("Model type could not be determined from model config.")
        return

    if model_type not in MODEL_TYPE_TO_APPLY_FN.keys():
        warnings.warn(f"No patch supported for model type: {model_type}.")
        return

    apply_fn = MODEL_TYPE_TO_APPLY_FN[model_type]

    if patch_locations:
        if isinstance(patch_locations, dict):
            locations_kwargs = {loc: True for loc in patch_locations if patch_locations[loc] == True}
        elif isinstance(patch_locations, Iterable):
            locations_kwargs = {loc: True for loc in patch_locations}
        else:
            raise TypeError("Invalid type of `patch_locations`, must be `Iterable` or `None`.")
        print(f"Applying patch to {model_type} model in: {tuple(locations_kwargs.keys())}")
        apply_fn(model=model, **locations_kwargs, compress_kwargs=compress_kwargs)
