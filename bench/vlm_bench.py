"""Expensive-stage cost: what one VLM look at one frame actually costs.

This is the denominator of every speedup claim in the project, so it is measured
at several visual-token budgets rather than assumed at one. `max_pixels` is the
knob that sets Qwen2.5-VL's per-frame cost; quoting "VLM frames/sec" without it
is meaningless.

Outputs GPU-seconds per frame, and what that implies for the two baselines:
  (a) dense VLM over 24 h of video at 1 fps  = 86,400 frames
  (b) uniform sampling at N frames           = the thing people actually do
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from framesieve.vlm import QwenYesNoScorer  # noqa: E402

FRAMES_PER_DAY_AT_1FPS = 86_400


@dataclass
class VlmResult:
    model: str
    max_visual_tokens: int
    batch: int
    attn: str
    frames_per_s: float
    s_per_frame: float
    peak_mem_gb: float
    gpu_hours_per_24h_at_1fps: float


def load_frames(paths: list[str], n: int) -> list[np.ndarray]:
    imgs = [np.array(Image.open(p).convert("RGB")) for p in paths]
    return [imgs[i % len(imgs)] for i in range(n)]


def bench(scorer: QwenYesNoScorer, frames: list[np.ndarray], question: str,
          batch: int, iters: int, warmup: int) -> tuple[float, float]:
    pool = [frames[i % len(frames)] for i in range(batch)]
    for _ in range(warmup):
        scorer.score(pool, question)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        scorer.score(pool, question)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return batch / dt, torch.cuda.max_memory_allocated() / 1e9


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-vl-7b")
    ap.add_argument("--frame-dir", default="/tmp/fs_frames")
    ap.add_argument("--token-budgets", type=int, nargs="*", default=[64, 128, 256, 512],
                    help="max visual tokens per frame")
    ap.add_argument("--batches", type=int, nargs="*", default=[1, 4, 16, 32])
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--question", default="Is there a stone bridge over the railway track?")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    paths = sorted(os.path.join(args.frame_dir, f)
                   for f in os.listdir(args.frame_dir) if f.endswith(".jpg"))
    frames = load_frames(paths, max(args.batches))
    print(f"{len(paths)} distinct source frames, "
          f"{frames[0].shape[1]}x{frames[0].shape[0]}\n")

    rows: list[VlmResult] = []
    for tb in args.token_budgets:
        px = tb * 28 * 28 * 4          # tokens -> pixels, accounting for 2x2 merge
        scorer = QwenYesNoScorer(args.model, max_pixels=px,
                                 min_pixels=min(px, 64 * 28 * 28))
        d = scorer.describe()
        print(f"=== {d['key']}  {d['params_b']}B  max_visual_tokens={d['max_visual_tokens']}"
              f"  attn={d['attn']}")
        for b in args.batches:
            try:
                fps, mem = bench(scorer, frames, args.question, b, args.iters, args.warmup)
            except torch.cuda.OutOfMemoryError:
                print(f"  batch {b:<4} OOM")
                torch.cuda.empty_cache()
                continue
            gh = FRAMES_PER_DAY_AT_1FPS / fps / 3600
            rows.append(VlmResult(model=args.model, max_visual_tokens=d["max_visual_tokens"],
                                  batch=b, attn=d["attn"], frames_per_s=fps,
                                  s_per_frame=1 / fps, peak_mem_gb=mem,
                                  gpu_hours_per_24h_at_1fps=gh))
            print(f"  batch {b:<4} {fps:>8.2f} frame/s  {1000/fps:>8.1f} ms/frame  "
                  f"{mem:>6.2f} GB   dense 24h: {gh:>7.2f} GPU-hours")
        del scorer
        torch.cuda.empty_cache()
        print()

    print(f"{'tokens/frame':>13}{'batch':>7}{'frame/s':>10}{'ms/frame':>11}"
          f"{'dense 24h (GPU-h)':>20}")
    print("-" * 62)
    for r in sorted(rows, key=lambda r: (r.max_visual_tokens, r.batch)):
        print(f"{r.max_visual_tokens:>13}{r.batch:>7}{r.frames_per_s:>10.2f}"
              f"{1000*r.s_per_frame:>11.1f}{r.gpu_hours_per_24h_at_1fps:>20.2f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"config": vars(args), "gpu": torch.cuda.get_device_name(0),
                       "results": [asdict(r) for r in rows]}, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
