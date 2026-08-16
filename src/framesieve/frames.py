"""Streaming frame source.

Decode is cheap (see runs/decode_*.json) but only if you never ask it to hand
every frame to Python. This pulls frames at a target rate through a single
ffmpeg pipe, in batches, as uint8 NHWC -- which is exactly what the encoder
wants and costs nothing to produce.

The decoder does the downscale, so the bytes crossing the pipe are already at
model input size: at 1 fps and 256x256, a 24 h video is 17 GB through the pipe
rather than 4.5 TB.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

_PTS_RE = re.compile(rb"pts_time:\s*([0-9.]+)")


@dataclass
class SourceInfo:
    path: str
    duration_s: float
    fps: float
    width: int
    height: int
    codec: str
    n_frames_est: int


def probe_source(path: str) -> SourceInfo:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames",
         "-show_entries", "format=duration", "-of", "json", path],
        capture_output=True, text=True, check=True)
    d = json.loads(out.stdout)
    st, fmt = d["streams"][0], d["format"]

    def _rate(s):
        if not s or s == "0/0":
            return 0.0
        a, _, b = s.partition("/")
        return float(a) / float(b) if b and float(b) else float(a)

    fps = _rate(st.get("avg_frame_rate")) or _rate(st.get("r_frame_rate"))
    dur = float(fmt.get("duration", 0.0))
    nb = st.get("nb_frames")
    return SourceInfo(path=path, duration_s=dur, fps=fps, width=int(st["width"]),
                      height=int(st["height"]), codec=st["codec_name"],
                      n_frames_est=int(nb) if nb and nb.isdigit() else int(dur * fps))


class FrameStream:
    """Yields (timestamps, frames_uint8) batches at `target_fps`.

    Timestamps are the frames' *true* presentation timestamps, read back from
    ffmpeg via the `showinfo` filter, not `start + k/rate`.

    That distinction is not pedantry. The `fps` filter resamples by picking the
    source frame nearest each output slot, so on a 25 fps source sampled at 1 fps
    the frame you get can sit up to half a second away from where you assumed it
    was. On moving footage half a second is a completely different picture, and
    anything that later re-fetches "the frame at t" gets a different image than
    the one that was indexed. scripts/verify.py checks the two paths agree.
    """

    def __init__(self, path: str, target_fps: float = 1.0, size=256,
                 batch: int = 256, start_s: float = 0.0, duration_s: float = 0.0,
                 gpu: bool = False, threads: int | None = None):
        """`size` is either an int (square, what SigLIP wants) or a (w, h) pair,
        or None to keep the source resolution. The VLM stage wants the source
        aspect ratio -- squashing 960x720 into 720x720 is a silent quality loss."""
        self.info = probe_source(path)
        self.path = path
        self.target_fps = target_fps
        if size is None:
            self.out_w, self.out_h = self.info.width, self.info.height
        elif isinstance(size, int):
            self.out_w = self.out_h = size
        else:
            self.out_w, self.out_h = int(size[0]), int(size[1])
        self.size = self.out_w  # backwards-compatible alias for square use
        self.batch = batch
        self.start_s = start_s
        self.duration_s = duration_s or max(0.0, self.info.duration_s - start_s)
        self.gpu = gpu
        self.threads = threads or min(32, os.cpu_count() or 8)
        self.n_expected = int(self.duration_s * target_fps)
        self.pts_exact = True          # cleared if showinfo ever falls behind
        self.n_pts_seen = 0

    def _cmd(self) -> list[str]:
        # showinfo logs at INFO level, so -v error would silently discard the very
        # timestamps we need
        cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-v", "info", "-nostats"]
        if self.start_s > 0:
            cmd += ["-ss", f"{self.start_s}"]
        if self.gpu:
            cmd += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
                    "-c:v", "h264_cuvid"]
        else:
            cmd += ["-threads", str(self.threads)]
        cmd += ["-i", self.path]
        if self.duration_s > 0:
            cmd += ["-t", f"{self.duration_s}"]
        cmd += ["-an", "-sn"]
        # `select` rather than `fps=`: the fps filter re-stamps its output onto a
        # nominal grid, so its timestamps do not identify the source frame it
        # chose, and a later seek to "the frame at t" lands on a different
        # picture. select keeps the original PTS, which makes the index and the
        # refine stage refer to the same frame. Verified in scripts/verify.py.
        period = 1.0 / self.target_fps
        vf = [f"select='isnan(prev_selected_t)+gte(t-prev_selected_t\\,{period})'"]
        if self.gpu:
            vf += [f"scale_cuda={self.out_w}:{self.out_h}", "hwdownload", "format=nv12"]
        else:
            vf += [f"scale={self.out_w}:{self.out_h}"]
        # showinfo reports each surviving frame's true pts_time to stderr, in
        # output order, so the k-th line pairs with the k-th frame on stdout
        vf.append("showinfo")
        cmd += ["-vf", ",".join(vf), "-vsync", "0",
                "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
        return cmd

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        fb = self.out_w * self.out_h * 3
        cmd = self._cmd()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, bufsize=fb * 16)

        # drain stderr on a thread: showinfo is chatty enough to fill the pipe
        # buffer and deadlock the whole decode if nobody reads it
        pts: list[float] = []
        err_tail: list[bytes] = []

        def _drain():
            for line in iter(proc.stderr.readline, b""):
                m = _PTS_RE.search(line)
                if m:
                    pts.append(float(m.group(1)))
                elif len(err_tail) < 200:
                    err_tail.append(line)

        t = threading.Thread(target=_drain, daemon=True)
        t.start()

        idx = 0
        try:
            while True:
                want = self.batch * fb
                buf = proc.stdout.read(want)
                if not buf:
                    break
                n = len(buf) // fb
                if n == 0:
                    break
                arr = np.frombuffer(buf[: n * fb], dtype=np.uint8).reshape(
                    n, self.out_h, self.out_w, 3)

                # showinfo runs ahead of the muxer, so the timestamps are almost
                # always already here; wait briefly if they are not
                deadline = time.monotonic() + 10.0
                while len(pts) < idx + n and time.monotonic() < deadline:
                    time.sleep(0.002)
                if len(pts) >= idx + n:
                    ts = np.array(pts[idx: idx + n], dtype=np.float64) + self.start_s
                else:
                    # last resort: nominal grid, and say so rather than pretend
                    ts = self.start_s + (np.arange(idx, idx + n) / self.target_fps)
                    self.pts_exact = False
                idx += n
                yield ts, arr
                if len(buf) < want:
                    break
        finally:
            try:
                proc.stdout.close()
            except Exception:
                pass
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except Exception:
                proc.kill()
            t.join(timeout=5)          # the drain thread owns stderr, not us
            try:
                proc.stderr.close()
            except Exception:
                pass
            self.n_pts_seen = len(pts)
            if idx == 0:
                raise RuntimeError(
                    "no frames decoded:\n"
                    + b"".join(err_tail[-20:]).decode("utf8", "ignore")[-1500:])
