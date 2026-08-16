"""Baseline (a): dense VLM over every sampled frame. Also the project's ground truth.

This is the quality ceiling and the honest cost of the thing everyone says is too
expensive. It scores every frame of the video at `--fps` against every query, at
the VLM's native visual-token budget, and writes the raw scores.

Nothing downstream is allowed to be more accurate than this file, and every
speedup in the repo is measured against its cost.

It checkpoints every `--checkpoint-every` batches: a multi-hour run that dies at
90% should not cost 90% of a GPU-day.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from framesieve.frames import FrameStream, probe_source  # noqa: E402
from framesieve.vlm import QwenYesNoScorer  # noqa: E402

# Chosen to span a wide range of rarity, from "happens constantly" to "may never
# happen". Written as yes/no questions because that is what the scorer answers.
DEFAULT_QUERIES = [
    "Is the train inside a tunnel?",
    "Is there another train visible?",
    "Is there a level crossing where a road crosses the railway?",
    "Is there a railway station platform?",
    "Is there a lake, loch or large body of water?",
    "Is there a bridge or viaduct crossing over the railway?",
    "Are there sheep or cattle visible?",
    "Are there people visible?",
]


def env_report() -> dict:
    def _sh(c):
        try:
            return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
        except Exception:
            return "?"
    return {"platform": platform.platform(), "machine": platform.machine(),
            "python": sys.version.split()[0], "torch": torch.__version__,
            "gpu": _sh("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader"),
            "ffmpeg": _sh("ffmpeg -version 2>/dev/null | head -1"),
            "git_commit": _sh("git rev-parse --short HEAD")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/glasgow_mallaig.mp4")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--model", default="qwen2.5-vl-7b")
    ap.add_argument("--max-visual-tokens", type=int, default=256,
                    help="cap; the source resolution binds first if it is smaller")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint-every", type=int, default=20)
    ap.add_argument("--out", default="runs/groundtruth_glasgow.npz")
    ap.add_argument("--queries-file", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    queries = DEFAULT_QUERIES
    if args.queries_file:
        with open(args.queries_file) as f:
            queries = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    info = probe_source(args.video)
    px = args.max_visual_tokens * 28 * 28 * 4
    scorer = QwenYesNoScorer(args.model, max_pixels=px,
                             min_pixels=min(px, 64 * 28 * 28))
    desc = scorer.describe()
    print(json.dumps(desc, indent=2))
    print(f"video: {info.width}x{info.height} {info.duration_s/3600:.2f} h")
    print(f"actual visual tokens/frame: "
          f"{scorer.visual_tokens_per_frame((info.height, info.width))}")
    print(f"{len(queries)} queries x ~{int(info.duration_s*args.fps):,} frames\n")

    stream = FrameStream(args.video, target_fps=args.fps, size=None,
                         batch=args.batch, start_s=args.start,
                         duration_s=args.duration)

    all_ts: list[np.ndarray] = []
    all_sc: list[np.ndarray] = []          # (n_frames, n_queries)
    t0 = time.perf_counter()
    n_done = 0
    n_batches = 0

    def checkpoint():
        ts = np.concatenate(all_ts) if all_ts else np.zeros(0, np.float32)
        sc = np.concatenate(all_sc) if all_sc else np.zeros((0, len(queries)), np.float32)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        np.savez_compressed(
            args.out, ts=ts.astype(np.float32), scores=sc.astype(np.float32),
            queries=np.array(queries, dtype=object),
            meta=json.dumps({"config": vars(args), "model": desc,
                             "env": env_report(),
                             "video": {"path": os.path.abspath(args.video),
                                       "width": info.width, "height": info.height,
                                       "duration_s": info.duration_s,
                                       "codec": info.codec},
                             "elapsed_s": time.perf_counter() - t0,
                             "n_frames_done": int(len(ts))}),
            allow_pickle=True)

    for ts, frames in stream:
        sc = np.stack([scorer.score(list(frames), q) for q in queries], axis=1)
        all_ts.append(ts.astype(np.float32))
        all_sc.append(sc.astype(np.float32))
        n_done += len(frames)
        n_batches += 1
        if n_batches % 5 == 0:
            el = time.perf_counter() - t0
            rate = n_done / el
            eta = (stream.n_expected - n_done) / rate if rate else 0
            print(f"  {n_done:>7,}/{stream.n_expected:,} frames  "
                  f"{rate:>5.2f} frame/s  {rate*len(queries):>6.1f} scores/s  "
                  f"elapsed {el/60:>6.1f} min  eta {eta/60:>6.1f} min", flush=True)
        if n_batches % args.checkpoint_every == 0:
            checkpoint()

    checkpoint()
    el = time.perf_counter() - t0
    print(f"\ndone: {n_done:,} frames x {len(queries)} queries = "
          f"{n_done*len(queries):,} VLM scores in {el/3600:.2f} GPU-hours")
    print(f"cost of a dense VLM pass over this video, per query: "
          f"{el/len(queries)/3600:.3f} GPU-hours")
    print(f"extrapolated to 24 h of video at {args.fps} fps, per query: "
          f"{el/len(queries)/3600 * (86400*args.fps)/n_done:.2f} GPU-hours")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
