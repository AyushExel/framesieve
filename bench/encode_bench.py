"""Dense-pass encoder throughput.

Decode turned out not to be the bottleneck (see runs/decode_*.json), so the cost
of the cheap dense pass is set by the vision encoder. This measures how many
frames per second each candidate encoder can absorb, and converts that into the
only unit anyone actually cares about: how long it takes to index a day of video.

Measured, not estimated:
  - throughput is timed with CUDA events around a steady-state loop, after warmup
  - the uint8 -> normalised-tensor preprocessing is inside the timed region,
    because that cost is real and is often quietly excluded
  - peak memory is recorded so the batch sizes are honest about what they need
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from framesieve.encoders import SIGLIP_MODELS, SiglipEncoder  # noqa: E402

FRAMES_PER_DAY_AT_1FPS = 86_400


@dataclass
class EncResult:
    model: str
    input_size: int
    params_vision_m: float
    batch: int
    dtype: str
    compiled: bool
    frames_per_s: float
    ms_per_batch: float
    peak_mem_gb: float
    hours_per_24h_at_1fps: float
    gpu_s_per_24h_at_1fps: float


def bench_one(enc: SiglipEncoder, batch: int, src_hw=(720, 960), iters: int = 30,
              warmup: int = 8) -> tuple[float, float, float]:
    """Return (frames_per_s, ms_per_batch, peak_mem_gb).

    Frames are generated once on the host as uint8 NHWC -- the same layout a
    decoder produces -- and the host->device copy plus preprocessing is timed.
    """
    H, W = src_hw
    host = torch.randint(0, 255, (batch, H, W, 3), dtype=torch.uint8).pin_memory()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        enc.encode_frames(host)
    torch.cuda.synchronize()

    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        enc.encode_frames(host)
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / iters
    return batch / (ms / 1000.0), ms, torch.cuda.max_memory_allocated() / 1e9


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["siglip2-base-224", "siglip2-base-384",
                                                    "siglip2-so400m-384"])
    ap.add_argument("--batches", type=int, nargs="*", default=[32, 64, 128, 256])
    ap.add_argument("--src-hw", type=int, nargs=2, default=(720, 960))
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"source frames: {args.src_hw[1]}x{args.src_hw[0]} uint8 NHWC (pinned host)")
    print(f"'24h index' = {FRAMES_PER_DAY_AT_1FPS:,} frames (24 h sampled at 1 fps)\n")

    rows: list[EncResult] = []
    for key in args.models:
        if key not in SIGLIP_MODELS:
            print(f"skip unknown model {key}")
            continue
        enc = SiglipEncoder(key, compile_model=args.compile)
        d = enc.describe()
        print(f"=== {key}  vision={d['params_vision_m']}M  in={d['input_size']}  "
              f"dim={d['embed_dim']}  {d['dtype']}"
              + ("  [compiled]" if args.compile else ""))
        for b in args.batches:
            try:
                fps, ms, mem = bench_one(enc, b, tuple(args.src_hw), args.iters)
            except torch.cuda.OutOfMemoryError:
                print(f"  batch {b:<5} OOM")
                torch.cuda.empty_cache()
                continue
            gpu_s = FRAMES_PER_DAY_AT_1FPS / fps
            r = EncResult(model=key, input_size=d["input_size"],
                          params_vision_m=d["params_vision_m"], batch=b,
                          dtype=str(enc.dtype), compiled=args.compile,
                          frames_per_s=fps, ms_per_batch=ms, peak_mem_gb=mem,
                          hours_per_24h_at_1fps=gpu_s / 3600.0,
                          gpu_s_per_24h_at_1fps=gpu_s)
            rows.append(r)
            print(f"  batch {b:<5} {fps:>9.1f} frame/s  {ms:>8.1f} ms/batch  "
                  f"{mem:>6.2f} GB peak   24h index: {gpu_s/60:>7.1f} min")
        del enc
        torch.cuda.empty_cache()
        print()

    print(f"{'model':<22}{'batch':>6}{'frame/s':>11}{'GB':>7}{'24h index (min)':>18}")
    print("-" * 64)
    for r in sorted(rows, key=lambda r: -r.frames_per_s):
        print(f"{r.model:<22}{r.batch:>6}{r.frames_per_s:>11.1f}{r.peak_mem_gb:>7.2f}"
              f"{r.gpu_s_per_24h_at_1fps/60:>18.1f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"config": vars(args),
                       "gpu": torch.cuda.get_device_name(0),
                       "torch": torch.__version__,
                       "results": [asdict(r) for r in rows]}, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
