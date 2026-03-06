from collections import OrderedDict


class Module:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class Sequential(Module):
    def __init__(self, *modules):
        self._modules = OrderedDict()
        for i, module in enumerate(modules):
            if isinstance(module, tuple) and len(module) == 2:
                name, obj = module
            else:
                name, obj = str(i), module
            self._modules[str(name)] = obj

    def add_module(self, name, module):
        self._modules[str(name)] = module

    def forward(self, x):
        out = x
        for module in self._modules.values():
            out = module(out)
        return out

    def named_modules(self):
        return list(self._modules.items())
