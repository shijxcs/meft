from collections import UserDict
from typing import *
from weakref import ref, WeakKeyDictionary

import torch

torch.compiler.allow_in_graph(ref)


class HashKeyDict(UserDict):
    def __contains__(self, key):
        for k in self.data.keys():
            if hash(k) == hash(key):
                return True
        return False

    def __getitem__(self, key):
        for k in self.data.keys():
            if hash(k) == hash(key):
                return self.data[k]
        raise KeyError(key)

    def __setitem__(self, key, item):
        for k in self.data.keys():
            if hash(k) == hash(key):
                del self.data[k]
                break
        self.data[key] = item

    def __delitem__(self, key):
        for k in self.data.keys():
            if hash(k) == hash(key):
                del self.data[k]
                break
        raise KeyError(key)


class WeakHashKeyDictionary(WeakKeyDictionary):
    """
    This class is to resolve the problem of `weakref.WeakKeyDictionary` class.
    The dict key matching first checks whether the keys are the same object (`a is b`);
    if not, it further checks whether the keys are equal (`a == b`).

    When using `weakref` with callback mechanism as dict key, it may cause some mismatching errors,
    because `ref(key) is ref(key)` returns `True`, but `ref(key, callback) is ref(key, callback)` returns `False`.
    Furthermore, the comparison `ref(key, callback) == ref(key, callback)` must return a bool value,
    otherwise the dict key matching will result in an error;
    however, if key type has a custom __eq__ function, it will return the custom type.
    For example, if key is a tensor, it will return a bool tensor with the same shape.
    
    Comparing hash values are more efficient and stable, since `hash(ref(key)) == hash(ref(key)`
    and `hash(ref(key, callback)) == hash(ref(key, callback)` both return bool values.
    """
    def __init__(self, dict=None):
        super().__init__(dict)
        self.data = HashKeyDict(self.data)
