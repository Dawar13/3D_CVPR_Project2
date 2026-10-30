"""Stub of basicsr.utils.registry: DFRSC's network file only needs ARCH_REGISTRY.register() as a decorator."""


class _Reg(dict):
    def register(self, obj=None):
        def deco(o):
            self[o.__name__] = o
            return o
        return deco(obj) if obj is not None else deco


ARCH_REGISTRY = _Reg()
DATASET_REGISTRY = _Reg()
MODEL_REGISTRY = _Reg()
