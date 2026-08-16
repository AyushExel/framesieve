"""Build the cheap dense index, with temporal redundancy collapse.

Two independent kinds of collapse live here, and they save different things:

  pixel gate (before the encoder)
      A 32x32 grayscale difference against the last encoded frame. If the frame
      has barely changed, reuse the previous embedding and never run the encoder.
      This is what makes static footage cost near zero *to index*.

  segment merge (after the encoder)
      Walk frames in time order and start a new segment whenever the embedding
      stops resembling the current segment's centroid. This is what makes static
      footage cost near zero *to search*: it turns "top-100 frames" -- which on
      real video is usually 100 near-identical frames from one moment -- into
      "top-100 distinct moments".

The second one matters far more than the first, and the ablation in this repo
shows why: the encoder is already so cheap that skipping it saves little, but
candidate diversity is worth a great deal at fixed VLM budget.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np

from .frames import FrameStream, probe_source

if TYPE_CHECKING:                      # pragma: no cover
    from .encoders import SiglipEncoder

# torch and the encoder are only needed to BUILD an index, never to read one.
# Keeping them out of module scope means `framesieve.load(...)` works on a
# machine with no GPU stack installed at all, which is most machines that only
# want to query an index someone else built.


@dataclass
class IndexStats:
    video: str
    duration_s: float
    target_fps: float
    n_frames: int
    n_encoded: int
    n_segments: int
    encoder: str
    encoder_revision: str
    embed_dim: int
    pixel_gate_tau: float
    segment_tau: float
    decode_encode_s: float
    frames_per_s: float
    realtime_factor: float
    gpu: str = ""
    seed: int = 0

    def pretty(self) -> str:
        skip = 100.0 * (1 - self.n_encoded / max(1, self.n_frames))
        return (f"{self.n_frames:,} frames @ {self.target_fps} fps -> "
                f"{self.n_encoded:,} encoded ({skip:.1f}% skipped by pixel gate) -> "
                f"{self.n_segments:,} segments\n"
                f"  {self.decode_encode_s:.1f} s wall, {self.frames_per_s:.0f} frame/s, "
                f"{self.realtime_factor:.0f}x realtime")


class FrameIndex:
    """Timestamps, embeddings and a segmentation over them."""

    def __init__(self, ts: np.ndarray, emb: np.ndarray, seg_id: np.ndarray,
                 stats: IndexStats):
        self.ts = ts.astype(np.float32)
        self.emb = emb.astype(np.float16)
        self.seg_id = seg_id.astype(np.int32)
        self.stats = stats
        self._seg_cache: tuple | None = None
        self._adj: np.ndarray | None = None

    # -- segments ----------------------------------------------------------

    def segments(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(seg_start_idx, seg_end_idx_exclusive, t_start, t_end) per segment."""
        if self._seg_cache is None:
            bounds = np.flatnonzero(np.diff(self.seg_id)) + 1
            starts = np.concatenate([[0], bounds])
            ends = np.concatenate([bounds, [len(self.seg_id)]])
            self._seg_cache = (starts, ends, self.ts[starts], self.ts[ends - 1])
        return self._seg_cache

    def adjacent_similarity(self) -> np.ndarray:
        """Cosine similarity between consecutive frames, cached.

        This is the raw material for cutting the video into any number of
        segments you like: the biggest drops are the moments the picture
        changed most. Computing it once means the segmentation granularity can
        become a *query-time* decision rather than an index-time one.
        """
        if getattr(self, "_adj", None) is None:
            e = self.emb.astype(np.float32)
            self._adj = np.einsum("ij,ij->i", e[:-1], e[1:])
        return self._adj

    def cut_into(self, n_segments: int) -> np.ndarray:
        """Segment ids for exactly `n_segments` segments, cutting at the biggest
        frame-to-frame changes. Deterministic and O(N log N)."""
        n = len(self.ts)
        k = int(np.clip(n_segments, 1, n))
        if k <= 1:
            return np.zeros(n, np.int32)
        adj = self.adjacent_similarity()
        if k - 1 >= len(adj):
            # asking for as many segments as frames: every gap is a cut
            cuts = np.arange(len(adj)) + 1
        else:
            cuts = np.sort(np.argpartition(adj, k - 1)[: k - 1]) + 1
        seg = np.zeros(n, np.int32)
        seg[cuts] = 1
        return np.cumsum(seg).astype(np.int32)

    def segment_reps(self) -> np.ndarray:
        """L2-normalised mean embedding per segment."""
        starts, ends, _, _ = self.segments()
        e = self.emb.astype(np.float32)
        reps = np.stack([e[s:t].mean(0) for s, t in zip(starts, ends)])
        return reps / (np.linalg.norm(reps, axis=1, keepdims=True) + 1e-8)

    # -- io ----------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez_compressed(path, ts=self.ts, emb=self.emb, seg_id=self.seg_id,
                            stats=json.dumps(asdict(self.stats)))

    @classmethod
    def load(cls, path: str) -> FrameIndex:
        z = np.load(path, allow_pickle=False)
        stats = IndexStats(**json.loads(str(z["stats"])))
        return cls(z["ts"], z["emb"], z["seg_id"], stats)


# --------------------------------------------------------------------------


def _gray_small(frames: np.ndarray, k: int = 32) -> np.ndarray:
    """Cheap 32x32 grayscale thumbnails for the pixel gate, on CPU, uint8 in."""
    n, h, w, _ = frames.shape
    sh, sw = max(1, h // k), max(1, w // k)
    g = frames[:, : sh * k, : sw * k, :].astype(np.float32).mean(axis=3)
    return g.reshape(n, k, sh, k, sw).mean(axis=(2, 4))


def build_index(video: str, encoder: SiglipEncoder, *, target_fps: float = 1.0,
                batch: int = 256, size: int = 256, pixel_gate_tau: float = 0.0,
                segment_tau: float = 0.0, start_s: float = 0.0,
                duration_s: float = 0.0, gpu_decode: bool = False,
                seed: int = 0, verbose: bool = True) -> FrameIndex:
    """Decode -> (optional pixel gate) -> encode -> (optional segment merge).

    pixel_gate_tau : mean absolute 32x32 grayscale difference below which a frame
                     is considered a repeat of the last encoded one. 0 disables.
                     Units are 0-255 grey levels; 2.0 is "visually identical".
    segment_tau    : cosine similarity to the running segment centroid below
                     which a new segment starts. 0 disables (every frame its own
                     segment).
    """
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)

    info = probe_source(video)
    stream = FrameStream(video, target_fps=target_fps, size=size, batch=batch,
                         start_s=start_s, duration_s=duration_s, gpu=gpu_decode)

    ts_all: list[np.ndarray] = []
    emb_all: list[np.ndarray] = []
    n_frames = n_encoded = 0
    last_thumb: np.ndarray | None = None
    last_emb: np.ndarray | None = None

    t0 = time.perf_counter()
    for ts, frames in stream:
        n_frames += len(frames)

        if pixel_gate_tau > 0:
            thumbs = _gray_small(frames)
            keep = np.ones(len(frames), dtype=bool)
            ref = last_thumb
            for i in range(len(frames)):
                if ref is not None and np.abs(thumbs[i] - ref).mean() < pixel_gate_tau:
                    keep[i] = False
                else:
                    ref = thumbs[i]
            last_thumb = ref
        else:
            keep = np.ones(len(frames), dtype=bool)

        idx_keep = np.flatnonzero(keep)
        if len(idx_keep):
            sel = torch.from_numpy(np.ascontiguousarray(frames[idx_keep]))
            e_kept = encoder.encode_frames(sel).cpu().numpy().astype(np.float32)
            n_encoded += len(idx_keep)
        else:
            e_kept = np.zeros((0, encoder.spec.dim), dtype=np.float32)

        # fill skipped frames forward with the embedding they were judged equal to
        e_batch = np.empty((len(frames), e_kept.shape[1] if len(e_kept)
                            else (last_emb.shape[0] if last_emb is not None else 0)),
                           dtype=np.float32)
        j = 0
        for i in range(len(frames)):
            if keep[i]:
                e_batch[i] = e_kept[j]
                last_emb = e_kept[j]
                j += 1
            else:
                e_batch[i] = last_emb
        ts_all.append(ts.astype(np.float32))
        emb_all.append(e_batch)

    dt = time.perf_counter() - t0
    ts_cat = np.concatenate(ts_all) if ts_all else np.zeros(0, np.float32)
    emb_cat = np.concatenate(emb_all) if emb_all else np.zeros((0, encoder.spec.dim),
                                                               np.float32)

    seg_id = _segment(emb_cat, segment_tau)

    covered = duration_s or max(0.0, info.duration_s - start_s)
    stats = IndexStats(
        video=os.path.abspath(video), duration_s=covered, target_fps=target_fps,
        n_frames=n_frames, n_encoded=n_encoded, n_segments=int(seg_id.max() + 1) if len(seg_id) else 0,
        encoder=encoder.spec.key, encoder_revision=encoder.spec.revision,
        embed_dim=encoder.spec.dim, pixel_gate_tau=pixel_gate_tau,
        segment_tau=segment_tau, decode_encode_s=dt,
        frames_per_s=n_frames / dt if dt else 0.0,
        realtime_factor=covered / dt if dt else 0.0,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        seed=seed)
    idx = FrameIndex(ts_cat, emb_cat, seg_id, stats)
    if verbose:
        print(stats.pretty())
    return idx


def _segment(emb: np.ndarray, tau: float) -> np.ndarray:
    """Greedy streaming segmentation on cosine similarity to a running centroid."""
    n = len(emb)
    if n == 0:
        return np.zeros(0, np.int32)
    if tau <= 0:
        return np.arange(n, dtype=np.int32)

    seg = np.zeros(n, np.int32)
    centroid = emb[0].copy()
    count = 1
    cur = 0
    for i in range(1, n):
        c = centroid / (np.linalg.norm(centroid) + 1e-8)
        if float(c @ emb[i]) >= tau:
            centroid += emb[i]
            count += 1
        else:
            cur += 1
            centroid = emb[i].copy()
            count = 1
        seg[i] = cur
    return seg
