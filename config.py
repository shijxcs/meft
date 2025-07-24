from dataclasses import dataclass, field
from typing import *


@dataclass
class MeftConfig:
    """
    This is the base configuration class to store the configuration of a `MeftTrainer`.
    """

    patch_locations: Optional[Union[str, Iterable]] = field(
        default=None,
        metadata={
            "help": "Patch applied locations. Can be `str`, `Iterable`, or `NoneType`. "
            "If `str`, it should be `'layer'` or `'sublayer'`. "
            "`'layer'` indicates checkpointing each Transformer layer. "
            "`'sublayer'` indicates patching each norm layer and checkpointing each attn and mlp layers. "
            "If `Iterable`, it should be argument names of `meft.patch.apply_patch_to_*_model()`. "
            "If `NoneType`, no patch is applied."
        },
    )

    compress_kwargs: Optional[Mapping[str, Any]] = field(
        default=None,
        metadata={
            "help": "Key word arguments to be passed to `meft.compressed.CompressedTensor`."
            "If `NoneType`, no activation compression is applied."
        },
    )

    compress_workers: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of threads allocated for activation compression."
            "If `NoneType`, no thread is allocated."
            "Empirically, set to 2 can achieve sufficient efficiency."
        },
    )
