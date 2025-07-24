import warnings
from types import NoneType
from typing import *

import torch
from transformers import Trainer, TrainingArguments
from transformers.modeling_utils import PreTrainedModel

from .compressed import compress_processor
from .config import MeftConfig
from .patch import apply_patch_to_model


class MeftTrainerBase:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        try:
            i = cls.__bases__.index(MeftTrainerBase)
            cls.trainer_cls = cls.__bases__[i+1]
        except:
            raise TypeError(f"{cls.__name__} must inherit another class after MeftTrainerBase.")
        
    def __new__(cls, *args, **kwargs):
        if cls is MeftTrainerBase:
            raise TypeError(f"{cls.__name__} cannot be instantiated directly; it must be used as a parent class.")
        return super().__new__(cls)

    def __init__(
        self,
        model: Any = None,
        args: TrainingArguments = None,
        *trainer_args,
        meft_config: MeftConfig = None,
        **trainer_kwargs,
    ) -> None:
        if meft_config is None:
            meft_config = MeftConfig()

        self.meft_config = meft_config

        if args.gradient_checkpointing:
            warnings.warn(
                "The argument `gradient_checkpointing=True` is incompatible with `MeftTrainer`. "
                "Use `MeftTrainer(..., meft_config=MeftConfig(..., patch_locations='layer'))`."
            )

        self.trainer_cls.__init__(self, model, args, *trainer_args, **trainer_kwargs)

        if isinstance(meft_config.patch_locations, str):
            if meft_config.patch_locations == "layer":
                patch_locations = ("ckpt_layer",)
            elif meft_config.patch_locations == "sublayer":
                patch_locations = ("norm", "ckpt_attn", "ckpt_mlp")
            else:
                raise ValueError("Invalid value of `meft_config.patch_locations`, must be `'layer'` or `'sublayer'`.")
        elif isinstance(meft_config.patch_locations, (Iterable, NoneType)):
            patch_locations = meft_config.patch_locations
        else:
            raise TypeError("Invalid type of `meft_config.patch_locations`, must be `str`, `Iterable`, or `NoneType`.")

        if isinstance(meft_config.compress_kwargs, (dict, NoneType)):
            compress_kwargs = meft_config.compress_kwargs
        else:
            raise TypeError("Invalid type of `meft_config.compress_kwargs`, must be `dict`, or `NoneType`.")

        if compress_kwargs is not None and isinstance(meft_config.compress_workers, int):
            self.async_compress = True
            self.compress_workers = meft_config.compress_workers
        else:
            self.async_compress = False

        if isinstance(model, PreTrainedModel):
            apply_patch_to_model(self.model, patch_locations=patch_locations, compress_kwargs=compress_kwargs)
        elif hasattr(model, "get_base_model") and isinstance(model.get_base_model(), PreTrainedModel):
            apply_patch_to_model(self.model.get_base_model(), patch_locations=patch_locations, compress_kwargs=compress_kwargs)
        else:
            warnings.warn("The model is not an instance of PreTrainedModel. No patch will be applied.")

    def train(self, *args, **kwargs):
        if hasattr(self.model.config, "use_cache") and self.model.config.use_cache:
            # `use_cache=True` makes no sense when training.
            # See https://github.com/huggingface/transformers/issues/23808.
            print("Setting `use_cache=False` during training.")
            orig_use_cache = self.model.config.use_cache
            self.model.config.use_cache = False

        if self.async_compress:
            # Call any linear algebra function before running the concurrent tasks to initialise the linalg module.
            # See https://github.com/pytorch/pytorch/issues/90613.
            torch.inverse(torch.ones((1, 1), device="cuda"))

            def _hook_forward(module, *args):
                if module.training and compress_processor.running:
                    compress_processor.join()
            self.model.register_forward_hook(_hook_forward)
            compress_processor.start(num_workers=self.compress_workers)

        result = self.trainer_cls.train(self, *args, **kwargs)

        if compress_processor.running:
            compress_processor.stop()

        if "orig_use_cache" in locals():
            self.model.config.use_cache = orig_use_cache

        return result



class MeftTrainer:
    """
    Replace `Trainer` with `MeftTrainer` to support memory-efficint fine-tuning.
    For other types of trainers, please use `MeftTrainer[CustomTrainer]`.

    """
    def __class_getitem__(cls, trainer_cls):
        class MeftTrainerDerived(MeftTrainerBase, trainer_cls):
            def __init__(self, *args, **kwargs) -> None:
                MeftTrainerBase.__init__(self, *args, **kwargs)
        return MeftTrainerDerived

    def __new__(cls, *args, **kwargs):
        derived_class = cls.__class_getitem__(Trainer)
        return derived_class(*args, **kwargs)
