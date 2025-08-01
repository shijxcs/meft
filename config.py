from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass
class MeftConfig:
    """
    This is the base configuration class to store the configuration of a `MeftTrainer`.
    """

    patch_locations: str | Iterable | None = field(
        default=None,
        metadata={
            "help": "Patch applied locations. Can be `str`, `Iterable`, or `None`. "
            "If `str`, it should be `'layer'` or `'sublayer'`. "
            "`'layer'` indicates checkpointing each Transformer layer. "
            "`'sublayer'` indicates patching each norm layer and checkpointing each attn and mlp layers. "
            "If `Iterable`, it should be argument names of `meft.patch.apply_patch_to_*_model()`. "
            "If `None`, no patch is applied."
        },
    )

    compress_kwargs: dict | None = field(
        default=None,
        metadata={
            "help": "Key word arguments to be passed to `meft.compressed.CompressedTensor`."
            "If `None`, no activation compression is applied."
        },
    )

    compress_workers: int | None = field(
        default=None,
        metadata={
            "help": "Number of threads allocated for activation compression."
            "If `None`, no thread is allocated."
            "Empirically, set to 2 can achieve sufficient efficiency."
        },
    )
