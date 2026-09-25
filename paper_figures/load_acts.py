"""Restricted loader for the repo's committed Activations pickles (no repo import)."""
import pickle, torch, collections, numpy as np

class ActivationsStub:
    def __setstate__(self, st):
        self.__dict__.update(st)

ALLOWED = {
    ("torch._utils", "_rebuild_tensor_v2"), ("torch._utils", "_rebuild_parameter"),
    ("torch.storage", "_load_from_bytes"), ("collections", "OrderedDict"),
    ("collections", "defaultdict"), ("builtins", "set"), ("builtins", "list"), ("builtins","dict"),
    ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
    ("numpy", "ndarray"), ("numpy", "dtype"), ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "scalar"), ("torch", "FloatStorage"), ("torch", "BFloat16Storage"),
    ("torch", "HalfStorage"), ("torch", "float32"), ("torch", "bfloat16"), ("torch", "float16"),
}

class U(pickle.Unpickler):
    def find_class(self, mod, name):
        if (mod, name) == ("src.activations", "Activations"):
            return ActivationsStub
        if (mod, name) in ALLOWED:
            import importlib
            return getattr(importlib.import_module(mod), name)
        raise pickle.UnpicklingError(f"blocked {mod}.{name}")

def load(path):
    with open(path, "rb") as f:
        return U(f).load()
