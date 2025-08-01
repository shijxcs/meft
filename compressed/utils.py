from collections import defaultdict
from collections.abc import Callable

from torch import Tensor

from ..linalg import *
from ..utils.threading import TaskProcessor
from ..utils.weakref import WeakHashKeyDictionary


COMPRESS_FUNC_MAPPING: dict[str, dict[str, Callable[..., tuple[Tensor, ...]]]] = {
    "lowrank": {
        "rqb": randomized_qb,
        "tsvd": truncated_svd,
        "rsvd": randomized_svd,
        "nyssvd": nystrom_svd,
    },
}

RECONSTRUCT_FUNC_MAPPING: dict[str, dict[str, Callable[..., Tensor]]] = {
    "lowrank": {
        "rqb": qb_reconstruct,
        "tsvd": svd_reconstruct,
        "rsvd": svd_reconstruct,
        "nyssvd": svd_reconstruct,
    },
}


_compress_cache = defaultdict(WeakHashKeyDictionary)
_compress_processor = TaskProcessor()


def get_compress_cache():
    return _compress_cache

def get_compress_processor():
    return _compress_processor
