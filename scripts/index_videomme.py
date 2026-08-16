"""Build the cheap dense index for every Video-MME long-split video.

This is the amortised half of the cascade's cost: it happens once per video, not
once per query, and every question about that video reuses it. The script reports
the total so the blog can quote a real "cost to index 205 hours of video" rather
than an extrapolation from one clip.

Indexes are written per video so the job is resumable -- 300 videos is long
enough that losing the lot to one bad file would be annoying.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from framesieve.encoders import CLIP_MODELS, ClipEncoder, SiglipEncoder  # noqa: E402
from framesieve.index import build_index  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", default="data/vmme_long")
    ap.add_argument("--out-dir", default="runs/vmme_index")
    ap.add_argument("--encoder", default="siglip2-base-224")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--segment-tau", type=float, default=0.90,
                    help="fixed from the held-out cab-ride video; NOT tuned here")
    ap.add_argument("--pixel-gate-tau", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.video_dir, "*.mp4")))
    if args.limit:
        paths = paths[: args.limit]

    # the encoder key decides the family, so a second family can be indexed
    # without a second script -- the pooling result has to be checked
    # against something that is not SigLIP
    enc = (ClipEncoder if args.encoder in CLIP_MODELS
           else SiglipEncoder)(args.encoder)
    print(json.dumps(enc.describe(), indent=2))
    print(f"{len(paths)} videos -> {args.out_dir}\n")

    tot_video_s = tot_wall = 0.0
    tot_frames = tot_segments = 0
    done = skipped = failed = 0
    t0 = time.perf_counter()

    for i, p in enumerate(paths):
        vid = os.path.splitext(os.path.basename(p))[0]
        out = os.path.join(args.out_dir, f"{vid}.npz")
        if os.path.exists(out):
            skipped += 1
            continue
        try:
            idx = build_index(p, enc, target_fps=args.fps, batch=args.batch,
                              size=args.size, segment_tau=args.segment_tau,
                              pixel_gate_tau=args.pixel_gate_tau, seed=args.seed,
                              verbose=False)
            idx.save(out)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  [FAIL] {vid}: {type(e).__name__}: {str(e)[:120]}", flush=True)
            continue
        done += 1
        tot_video_s += idx.stats.duration_s
        tot_wall += idx.stats.decode_encode_s
        tot_frames += idx.stats.n_frames
        tot_segments += idx.stats.n_segments
        if done % 20 == 0:
            el = time.perf_counter() - t0
            print(f"  {i+1:>4}/{len(paths)}  {tot_frames:>9,} frames  "
                  f"{tot_video_s/3600:>6.1f} h video  {el/60:>6.1f} min wall  "
                  f"{tot_video_s/max(el,1e-9):>6.0f}x realtime", flush=True)

    el = time.perf_counter() - t0
    summary = {
        "n_videos_indexed": done, "n_skipped_existing": skipped, "n_failed": failed,
        "total_video_hours": tot_video_s / 3600,
        "total_frames": tot_frames, "total_segments": tot_segments,
        "collapse_ratio": tot_frames / max(1, tot_segments),
        "wall_s": el, "realtime_factor": tot_video_s / max(el, 1e-9),
        "encoder": enc.describe(), "config": vars(args),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    with open(os.path.join(args.out_dir, "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nindexed {done} videos ({tot_video_s/3600:.1f} h of video) in "
          f"{el/60:.1f} min = {summary['realtime_factor']:.0f}x realtime")
    print(f"{tot_frames:,} frames -> {tot_segments:,} segments "
          f"({summary['collapse_ratio']:.1f}x collapse)")
    if failed:
        print(f"{failed} videos failed")


if __name__ == "__main__":
    main()
