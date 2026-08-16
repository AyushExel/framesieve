"""Random access to frames by timestamp.

The refine stage needs a handful of arbitrary frames at full resolution, not a
stream. A naive implementation seeks once per frame and discovers that seeking
costs more than the VLM call it was feeding -- so this fetches in parallel and
measures itself.

`ffmpeg -ss T -i file` is an *input* seek: it jumps to the keyframe at or before
T and decodes forward to T. That is both accurate and cheap, and it is why the
cost per frame is roughly one GOP of decoding rather than one video of decoding.
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .frames import probe_source


class FrameFetcher:
    """Fetch frames at given timestamps, in parallel, as uint8 NHWC."""

    def __init__(self, path: str, size=None, workers: int = 16):
        self.path = path
        self.info = probe_source(path)
        if size is None:
            self.w, self.h = self.info.width, self.info.height
        elif isinstance(size, int):
            self.w = self.h = size
        else:
            self.w, self.h = int(size[0]), int(size[1])
        self.workers = workers

    def _one(self, t: float) -> np.ndarray | None:
        # Seek half a source-frame early and take the first frame that comes out.
        # Landing exactly on a frame's own PTS is a coin flip -- float rounding
        # decides whether ffmpeg returns that frame or the next one -- and being
        # one frame off silently breaks the guarantee that a re-fetched frame is
        # the frame that was indexed.
        half = 0.5 / self.info.fps if self.info.fps > 0 else 0.0
        seek = max(0.0, t - half)
        cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error",
               "-ss", f"{seek:.6f}", "-i", self.path,
               "-frames:v", "1", "-an", "-sn"]
        if (self.w, self.h) != (self.info.width, self.info.height):
            cmd += ["-vf", f"scale={self.w}:{self.h}"]
        cmd += ["-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
        p = subprocess.run(cmd, capture_output=True)
        need = self.w * self.h * 3
        if len(p.stdout) < need:
            return None
        return np.frombuffer(p.stdout[:need], dtype=np.uint8).reshape(self.h, self.w, 3)

    def fetch(self, timestamps: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        """Return (kept_timestamps, frames). Frames that fail to decode are dropped."""
        ts = list(timestamps)
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            out = list(ex.map(self._one, ts))
        keep = [(t, f) for t, f in zip(ts, out) if f is not None]
        if not keep:
            return np.zeros(0, np.float32), np.zeros((0, self.h, self.w, 3), np.uint8)
        return (np.array([k[0] for k in keep], dtype=np.float32),
                np.stack([k[1] for k in keep]))

    def benchmark(self, n: int = 64, seed: int = 0) -> dict:
        rng = np.random.default_rng(seed)
        ts = rng.uniform(0, max(1.0, self.info.duration_s - 1), size=n)
        t0 = time.perf_counter()
        got_ts, frames = self.fetch(ts)
        dt = time.perf_counter() - t0
        return {"n_requested": n, "n_returned": int(len(frames)),
                "wall_s": dt, "frames_per_s": len(frames) / dt if dt else 0.0,
                "ms_per_frame_effective": 1000 * dt / max(1, len(frames)),
                "workers": self.workers}
