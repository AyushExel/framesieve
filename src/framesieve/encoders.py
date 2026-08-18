"""Cheap dense encoder: the first stage of the cascade.

This is the model that has to look at *every* frame, so its throughput sets the
floor on indexing cost. Everything here is built to keep the GPU fed:

  - frames arrive as uint8 NHWC (that is what a decoder hands you)
  - resize and normalise happen on the GPU, not in PIL
  - the image tower runs in bf16 under inference_mode
  - text is encoded once per query and cached

Model revisions are pinned. An unpinned encoder makes every number in this repo
unreproducible the moment upstream repacks a checkpoint.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

# Pinned revisions, resolved 2026-08-15. Update deliberately, never implicitly.
SIGLIP_MODELS: dict[str, dict] = {
    "siglip-base-224": dict(
        repo="google/siglip-base-patch16-224", revision="7fd15f0689c7", size=224),
    "siglip2-base-224": dict(
        repo="google/siglip2-base-patch16-224", revision="75de2d55ec2d", size=224),
    "siglip2-base-384": dict(
        repo="google/siglip2-base-patch16-384", revision="f775b65a7976", size=384),
    "siglip-so400m-384": dict(
        repo="google/siglip-so400m-patch14-384", revision="9fdffc58afc9", size=384),
    "siglip2-so400m-384": dict(
        repo="google/siglip2-so400m-patch14-384", revision="e8e487298228", size=384),
}


def pick_device(device: str | None = None) -> str:
    """CUDA if there is one, MPS on Apple silicon, else CPU.

    Defaulting to "cuda" and letting torch raise is the wrong failure: someone
    without a GPU gets `RuntimeError: No CUDA GPUs are available` from three
    frames down a stack, which says nothing about what to do. The cheap encoder
    runs perfectly well on CPU -- about 35 frame/s on eight threads, which is
    35x realtime at 1 fps -- so falling back is the right default, not an error.
    """
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None \
            and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str, dtype: torch.dtype | None = None) -> torch.dtype:
    """bfloat16 on an accelerator, float32 on CPU.

    bfloat16 on CPU is emulated on most builds and measurably slower than
    float32, so following the GPU default there would be a pessimisation.
    """
    if dtype is not None:
        return dtype
    return torch.float32 if device == "cpu" else torch.bfloat16


@dataclass
class EncoderSpec:
    key: str
    repo: str
    revision: str
    size: int
    dim: int = 0


class SiglipEncoder:
    """Text-image encoder used for the dense pass and for candidate selection."""

    # SigLIP normalises to [-1, 1]; both mean and std are 0.5 for every channel.
    MEAN = 0.5
    STD = 0.5

    def __init__(self, key: str = "siglip2-base-224",
                 device: str | None = None,
                 dtype: torch.dtype | None = None,
                 compile_model: bool = False):
        from transformers import AutoModel, AutoTokenizer

        device = pick_device(device)
        dtype = pick_dtype(device, dtype)

        if key not in SIGLIP_MODELS:
            raise KeyError(f"unknown encoder {key!r}; have {list(SIGLIP_MODELS)}")
        cfg = SIGLIP_MODELS[key]
        self.spec = EncoderSpec(key=key, repo=cfg["repo"], revision=cfg["revision"],
                                size=cfg["size"])
        self.device = device
        self.dtype = dtype
        self.model = AutoModel.from_pretrained(
            cfg["repo"], revision=cfg["revision"], dtype=dtype,
        ).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["repo"], revision=cfg["revision"])
        self.spec.dim = int(self.model.config.text_config.hidden_size)
        self._compiled = False
        if compile_model:
            self.model.vision_model = torch.compile(self.model.vision_model)
            self._compiled = True

    # -- images ------------------------------------------------------------

    def preprocess(self, frames_u8: torch.Tensor) -> torch.Tensor:
        """uint8 NHWC on any device -> normalised NCHW on self.device.

        Resizing is done here rather than by the decoder when the decoder's output
        size does not match the model, so that the same index can be built from
        differently-sized sources.
        """
        if frames_u8.dtype != torch.uint8:
            raise TypeError(f"expected uint8 frames, got {frames_u8.dtype}")
        x = frames_u8.to(self.device, non_blocking=True)
        x = x.permute(0, 3, 1, 2).to(self.dtype).div_(255.0)
        s = self.spec.size
        if x.shape[-1] != s or x.shape[-2] != s:
            x = F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False,
                              antialias=True)
        return x.sub_(self.MEAN).div_(self.STD)

    @staticmethod
    def _as_tensor(out) -> torch.Tensor:
        """transformers has changed what get_*_features returns across versions:
        sometimes a tensor, sometimes a ModelOutput. Accept either."""
        if isinstance(out, torch.Tensor):
            return out
        for attr in ("pooler_output", "last_hidden_state", "image_embeds", "text_embeds"):
            v = getattr(out, attr, None)
            if isinstance(v, torch.Tensor):
                return v
        raise TypeError(f"cannot extract features from {type(out)}")

    @torch.inference_mode()
    def encode_frames(self, frames_u8: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        pixel_values = self.preprocess(frames_u8)
        feats = self._as_tensor(self.model.get_image_features(pixel_values=pixel_values))
        if normalize:
            feats = F.normalize(feats.float(), dim=-1)
        return feats

    # -- text --------------------------------------------------------------

    @torch.inference_mode()
    def encode_text(self, texts: Sequence[str], normalize: bool = True) -> torch.Tensor:
        # SigLIP was trained with padding to a fixed 64-token context; using
        # dynamic padding here measurably changes the embedding.
        tok = self.tokenizer(list(texts), padding="max_length", max_length=64,
                             truncation=True, return_tensors="pt").to(self.device)
        feats = self._as_tensor(self.model.get_text_features(**tok))
        if normalize:
            feats = F.normalize(feats.float(), dim=-1)
        return feats

    @property
    def logit_scale(self) -> float:
        return float(self.model.logit_scale.exp().item())

    @property
    def logit_bias(self) -> float:
        b = getattr(self.model, "logit_bias", None)
        return float(b.item()) if b is not None else 0.0

    def describe(self) -> dict:
        n = sum(p.numel() for p in self.model.parameters())
        nv = sum(p.numel() for p in self.model.vision_model.parameters())
        return {"key": self.spec.key, "repo": self.spec.repo,
                "revision": self.spec.revision, "input_size": self.spec.size,
                "embed_dim": self.spec.dim, "params_total_m": round(n / 1e6, 1),
                "params_vision_m": round(nv / 1e6, 1), "dtype": str(self.dtype),
                "compiled": self._compiled}


# A second encoder family, so the pooling result is not a statement about SigLIP.
# CLIP differs in the ways that could plausibly matter: a softmax contrastive
# loss rather than SigLIP's pairwise sigmoid, different training data, and a
# different image normalisation. If the pooling optimum moves the same way under
# both, it is a property of the task.
CLIP_MODELS: dict[str, dict] = {
    "clip-b32": dict(repo="openai/clip-vit-base-patch32", revision=None, size=224),
    "clip-laion-b32": dict(repo="laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
                           revision=None, size=224),
}


class ClipEncoder(SiglipEncoder):
    """OpenAI/LAION CLIP behind the same interface as SiglipEncoder.

    Only three things actually differ: the channel normalisation, the text
    padding convention (CLIP pads to the longest in the batch, and forcing a
    fixed 64 changes the embedding), and the model registry it looks in.
    """

    MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

    def __init__(self, key: str = "clip-b32",
                 device: str | None = None,
                 dtype: torch.dtype | None = None,
                 compile_model: bool = False):
        from transformers import AutoModel, AutoTokenizer

        device = pick_device(device)
        dtype = pick_dtype(device, dtype)

        if key not in CLIP_MODELS:
            raise KeyError(f"unknown encoder {key!r}; have {list(CLIP_MODELS)}")
        cfg = CLIP_MODELS[key]
        self.spec = EncoderSpec(key=key, repo=cfg["repo"],
                                revision=cfg["revision"] or "", size=cfg["size"])
        self.device, self.dtype = device, dtype
        self.model = AutoModel.from_pretrained(cfg["repo"], dtype=dtype
                                               ).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["repo"])
        self.spec.dim = int(self.model.config.text_config.hidden_size)
        self.MEAN = self.MEAN.to(device=device, dtype=dtype)
        self.STD = self.STD.to(device=device, dtype=dtype)
        self._compiled = False
        if compile_model:
            self.model.vision_model = torch.compile(self.model.vision_model)
            self._compiled = True

    @torch.inference_mode()
    def encode_text(self, texts, normalize: bool = True) -> torch.Tensor:
        tok = self.tokenizer(list(texts), padding=True, truncation=True,
                             max_length=77, return_tensors="pt").to(self.device)
        feats = self._as_tensor(self.model.get_text_features(**tok))
        return F.normalize(feats.float(), dim=-1) if normalize else feats

    @property
    def logit_bias(self) -> float:
        return 0.0
