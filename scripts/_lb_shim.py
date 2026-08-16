"""Import LanguageBind's model class without its video-loading stack.

The package pulls in decord (no aarch64 wheel) and pytorchvideo (which imports a
torchvision module removed after 0.16) purely to decode videos. We feed the model
tensors we decoded ourselves, so the whole chain is stubbed rather than fought.
"""
import importlib.machinery
import sys
import types

import torchvision.transforms.functional as _F

_ft = types.ModuleType("torchvision.transforms.functional_tensor")
_ft.__spec__ = importlib.machinery.ModuleSpec("torchvision.transforms.functional_tensor", None)
for _n in dir(_F):
    if not _n.startswith("_"):
        setattr(_ft, _n, getattr(_F, _n))
sys.modules["torchvision.transforms.functional_tensor"] = _ft

_d = types.ModuleType("decord")
_d.__spec__ = importlib.machinery.ModuleSpec("decord", None)
_d.VideoReader = object
_d.cpu = lambda *a, **k: None
_d.bridge = types.SimpleNamespace(set_bridge=lambda *a: None)
sys.modules["decord"] = _d
