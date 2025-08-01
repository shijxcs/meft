import warnings
from collections.abc import Iterable

import torch
from transformers.trainer import Trainer
from transformers.modeling_utils import PreTrainedModel

from .compressed import get_compress_processor
from .config import MeftConfig
from .patch import apply_patch_to_model


class MeftTrainer(Trainer):
    """
    Trainer for Memory-Efficient Fine-Tuning (MEFT) method.

    This class implements `MeftTrainer` as a subclass of `transformers.Trainer`.
    It can also inherit from other trainer variants using the subscript syntax `MeftTrainer[T]`.

    Example:

    ```python
    from datasets import load_dataset
    from meft import MeftConfig, MeftTrainer
    from trl import SFTTrainer

    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="train[:1%]")

    trainer = MeftTrainer[SFTTrainer](
        model="Qwen/Qwen3-0.6B-Base",
        train_dataset=dataset,
        meft_config=MeftConfig(
            patch_locations="layer",
            compress_kwargs={"rank": 128, "niter": 1},
        ),
    )
    trainer.train()
    ```

    Args:
        *trainer_args:
            &#45; arguments of inherited trainer.
        meft_config (MeftConfig or None, optional):
            &#45; the MEFT configuration used to patch the model. If None, the model is not patched.
            See `meft.MeftConfig` for detailed parameters.
            Default: `None`.
        **trainer_kwargs:
            &#45; keyword arguments of inherited trainer.
    """
    def __class_getitem__(cls, T):
        return type(f"MeftTrainer[{T.__name__}]", (MeftTrainer, T), {})

    def __init__(
        self,
        *trainer_args,
        meft_config: MeftConfig | None = None,
        **trainer_kwargs,
    ):
        if meft_config is None:
            meft_config = MeftConfig()

        self.meft_config = meft_config

        super().__init__(*trainer_args, **trainer_kwargs)

        if self.args.gradient_checkpointing:
            warnings.warn(
                "The argument `gradient_checkpointing=True` is incompatible with `MeftTrainer`. "
                "Use `MeftTrainer(..., meft_config=MeftConfig(..., patch_locations='layer'))`."
            )

        if isinstance(meft_config.patch_locations, str):
            if meft_config.patch_locations == "layer":
                self.patch_locations = ("ckpt_layer",)
            elif meft_config.patch_locations == "sublayer":
                self.patch_locations = ("norm", "ckpt_attn", "ckpt_mlp")
            else:
                raise ValueError("Invalid value of `meft_config.patch_locations`, must be `'layer'` or `'sublayer'`.")
        elif isinstance(meft_config.patch_locations, Iterable | None):
            self.patch_locations = meft_config.patch_locations
        else:
            raise TypeError("Invalid type of `meft_config.patch_locations`, must be `str`, `Iterable`, or `None`.")

        if isinstance(meft_config.compress_kwargs, dict | None):
            self.compress_kwargs = meft_config.compress_kwargs
        else:
            raise TypeError("Invalid type of `meft_config.compress_kwargs`, must be `dict` or `None`.")

        if isinstance(meft_config.compress_workers, int | None):
            self.compress_workers = meft_config.compress_workers
        else:
            raise TypeError("Invalid type of `meft_config.compress_workers`, must be `int` or `None`.")

        if self.patch_locations:
            if isinstance(self.model, PreTrainedModel):
                apply_patch_to_model(
                    self.model,
                    patch_locations=self.patch_locations,
                    compress_kwargs=self.compress_kwargs
                )
            elif hasattr(self.model, "get_base_model") and isinstance(self.model.get_base_model(), PreTrainedModel):
                apply_patch_to_model(
                    self.model.get_base_model(),
                    patch_locations=self.patch_locations,
                    compress_kwargs=self.compress_kwargs
                )
            else:
                warnings.warn("The model is not an instance of PreTrainedModel. No patch will be applied.")

    def train(self, *args, **kwargs):
        if hasattr(self.model, "config") and hasattr(self.model.config, "use_cache") and self.model.config.use_cache:
            # `use_cache=True` makes no sense when training.
            # See https://github.com/huggingface/transformers/issues/23808.
            print("Setting `use_cache=False` during training.")
            self.is_cache_closed = True
            self.original_cache = self.model.config.use_cache
            self.model.config.use_cache = False
        else:
            self.is_cache_closed = False

        if self.patch_locations and self.compress_kwargs and self.compress_workers:
            # Call any linear algebra function before running the concurrent tasks to initialise the linalg module.
            # See https://github.com/pytorch/pytorch/issues/90613.
            torch.inverse(torch.ones((1, 1), device="cuda"))

            compress_processor = get_compress_processor()
            compress_processor.start(num_workers=self.compress_workers)
            def _hook_forward(module, *args):
                if module.training and compress_processor.running:
                    compress_processor.join()
            self.model.register_forward_hook(_hook_forward)

            self.compress_processor = compress_processor
        else:
            self.compress_processor = None

        result = super().train(*args, **kwargs)

        if self.compress_processor and self.compress_processor.running:
            self.compress_processor.stop()

        if self.is_cache_closed:
            self.model.config.use_cache = self.original_cache

        return result
