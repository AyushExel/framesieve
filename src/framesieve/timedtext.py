"""Timed text over a video, whatever produced it.

Speech and on-screen text arrive by completely different routes -- one from
Whisper over the audio, the other from an OCR pass over the frames -- and end up
as exactly the same thing: spans of time with words attached, embedded so they
can be searched. So they share a container rather than having one each.

`kind` says where the words came from, and travels through to the `source` on
every hit, because "the caption said it" and "someone said it" are different
answers to a query and a caller should be able to tell them apart.

The embeddings here are from a sentence encoder, not from SigLIP. SigLIP's text
tower is trained to sit beside images and is a poor text-to-text matcher; this is
a text-to-text problem. It also means these scores are on a different scale from
the frame scores, and the search layer never ranks the two against each other.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

__all__ = ["TimedText", "TimedTextIndex", "TextEncoder", "DEFAULT_TEXT_ENCODER"]

DEFAULT_TEXT_ENCODER = "BAAI/bge-small-en-v1.5"

# bge wants the query side marked and the passage side bare; using the wrong one
# costs a couple of points and would look like a retrieval problem
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass(frozen=True)
class TimedText:
    start: float
    end: float
    text: str


class TimedTextIndex:
    """Timed text spans and their embeddings.

    `kind` is "speech" or "text"; it is stored, so a loaded index still knows
    what it holds.
    """

    def __init__(self, segments: list[TimedText], emb: np.ndarray,
                 kind: str = "text", meta: dict | None = None):
        self.segments = segments
        self.emb = np.asarray(emb, dtype=np.float32)
        self.kind = kind
        self.meta = meta or {}

    def __len__(self) -> int:
        return len(self.segments)

    def __repr__(self) -> str:
        mins = (self.segments[-1].end / 60) if self.segments else 0.0
        return f"<TimedTextIndex {self.kind} {len(self)} spans over {mins:.1f} min>"

    @property
    def starts(self) -> np.ndarray:
        return np.array([s.start for s in self.segments], dtype=np.float64)

    @property
    def texts(self) -> list[str]:
        return [s.text for s in self.segments]

    def save(self, path: str) -> None:
        import lance
        import pyarrow as pa

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        d = int(self.emb.shape[1]) if self.emb.ndim == 2 and len(self.emb) else 0
        table = pa.table({
            "start": pa.array([s.start for s in self.segments], pa.float64()),
            "end": pa.array([s.end for s in self.segments], pa.float64()),
            "text": pa.array([s.text for s in self.segments], pa.string()),
            "emb": pa.FixedSizeListArray.from_arrays(
                pa.array(np.ascontiguousarray(self.emb).reshape(-1)), d),
        })
        lance.write_dataset(table, path, mode="overwrite")
        with open(os.path.join(path, "framesieve.json"), "w") as f:
            json.dump({"kind": self.kind, "meta": self.meta}, f)

    @classmethod
    def load(cls, path: str) -> TimedTextIndex:
        import lance

        t = lance.dataset(path).to_table(columns=["start", "end", "text", "emb"])
        segs = [TimedText(float(a), float(b), c) for a, b, c in
                zip(t.column("start").to_pylist(), t.column("end").to_pylist(),
                    t.column("text").to_pylist())]
        emb = (np.stack(t.column("emb").to_numpy(zero_copy_only=False))
               if segs else np.zeros((0, 0), np.float32))
        with open(os.path.join(path, "framesieve.json")) as f:
            side = json.load(f)
        return cls(segs, emb, side.get("kind", "text"), side.get("meta", {}))


class TextEncoder:
    """The sentence encoder, used on both sides of a text match."""

    def __init__(self, name: str = DEFAULT_TEXT_ENCODER,
                 device: str | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        from .encoders import pick_device, pick_dtype

        self.device = pick_device(device)
        self.dtype = pick_dtype(self.device, None)
        self.name = name
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(
            name, dtype=self.dtype).to(self.device).eval()
        self._torch = torch

    def encode(self, texts, *, query: bool = False, batch: int = 128,
               maxlen: int = 512) -> np.ndarray:
        torch = self._torch
        prefix = BGE_QUERY_PREFIX if query else ""
        out = []
        for i in range(0, len(texts), batch):
            b = [prefix + t for t in texts[i:i + batch]]
            enc = self.tok(b, padding=True, truncation=True, max_length=maxlen,
                           return_tensors="pt").to(self.device)
            with torch.inference_mode():
                h = self.model(**enc).last_hidden_state[:, 0]   # bge uses CLS
                h = torch.nn.functional.normalize(h.float(), dim=-1)
            out.append(h.cpu().numpy().astype(np.float32))
        return (np.concatenate(out) if out
                else np.zeros((0, self.model.config.hidden_size), np.float32))


def embed_segments(segments: list[TimedText], kind: str, *,
                   text_encoder: str = DEFAULT_TEXT_ENCODER,
                   device: str | None = None, meta: dict | None = None
                   ) -> TimedTextIndex:
    enc = TextEncoder(text_encoder, device=device)
    emb = enc.encode([s.text for s in segments])
    del enc
    return TimedTextIndex(segments, emb, kind,
                          {**(meta or {}), "text_encoder": text_encoder,
                           "n_segments": len(segments)})
