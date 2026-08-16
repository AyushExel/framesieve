"""Why does averaging the top few frames beat taking the best one?

Replacing `max` with `mean of the top 3` over the frames in a chunk is worth
+2.5 R@1 on MomentSeeker and costs nothing. There are two competing explanations
and they are not the same claim:

  persistence   A real event lasts several seconds, so its evidence appears in
                several consecutive frames. A spurious match appears in one.
                Requiring k frames to agree is a *temporal* prior, and k should
                correspond to a fixed number of seconds.

  variance      max over n noisy samples is a biased estimator of chunk
                relevance, and the bias grows with n and with within-chunk
                variance. Averaging the top k is simply a lower-variance
                statistic. This has nothing to do with time, and k should scale
                with the number of frames n.

They are separable. Hold the frame rate fixed and vary the chunk length:

  - persistence predicts the optimal k stays at a fixed number of *seconds*
  - variance    predicts the optimal k stays a fixed *fraction of n*

At 1 fps, a 5 s chunk holds 5 frames and a 20 s chunk holds 20. If the best k is
3 in both, it is persistence. If it is ~1.5 and ~6, it is variance.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from framesieve.benchmarks.momentseeker import (  # noqa: E402
    gt_chunk_mask,
    load_queries,
    map_at_5_matched,
    recall_at_k,
)
from framesieve.index import FrameIndex  # noqa: E402


def chunks_of(duration_s: float, chunk_s: float) -> np.ndarray:
    n = max(1, int(np.ceil(duration_s / chunk_s)))
    st = np.arange(n, dtype=np.float64) * chunk_s
    return np.stack([st, np.minimum(st + chunk_s, duration_s)], axis=1)


def topk_scores(ts: np.ndarray, sims: np.ndarray, ch: np.ndarray, k: int,
                stride: int = 1) -> np.ndarray:
    """Mean of the k highest frame scores in each chunk. k >= len(chunk) is mean."""
    idx = np.clip(np.searchsorted(ch[:, 0], ts, "right") - 1, 0, len(ch) - 1)
    out = np.full(len(ch), -1e9, dtype=np.float64)
    order = np.argsort(idx, kind="stable")
    idx_s, sims_s = idx[order], sims[order]
    bounds = np.searchsorted(idx_s, np.arange(len(ch) + 1))
    for c in range(len(ch)):
        v = sims_s[bounds[c]:bounds[c + 1]]
        if v.size:
            out[c] = np.sort(v)[-min(k, v.size):].mean()
    return out


def sweep(queries, qemb, index_dir, chunk_s: float, ks, fps_div: int = 1,
          cache=None) -> dict:
    """Evaluate a range of k at one chunk length, optionally after thinning the
    index to a lower effective frame rate."""
    cache = {} if cache is None else cache
    acc = {k: ([], []) for k in ks}
    n_frames_per_chunk = []
    for n, q in enumerate(queries):
        if q.video_id not in cache:
            if len(cache) > 40:
                cache.clear()
            cache[q.video_id] = FrameIndex.from_npz(
                os.path.join(index_dir, f"{q.video_id}.npz"))
        idx = cache[q.video_id]
        ts, emb = idx.ts, idx.emb
        if fps_div > 1:
            ts, emb = ts[::fps_div], emb[::fps_div]
        ch = chunks_of(float(ts[-1]) + 1.0, chunk_s)
        n_frames_per_chunk.append(len(ts) / max(1, len(ch)))
        sims = emb.astype(np.float32) @ qemb[n]
        is_gt = gt_chunk_mask(ch, q.gt_intervals)
        for k in ks:
            cs = topk_scores(ts, sims, ch, k)
            ranked = np.argsort(-cs)
            acc[k][0].append(recall_at_k(ranked, is_gt, 1))
            acc[k][1].append(map_at_5_matched(ranked, ch, q.gt_intervals))
    return dict(
        chunk_s=chunk_s, fps_div=fps_div,
        frames_per_chunk=float(np.mean(n_frames_per_chunk)),
        by_k={str(k): dict(R1=float(np.mean(a)) * 100, mAP5=float(np.mean(b)) * 100)
              for k, (a, b) in acc.items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="data/ms_raw/t2v.json")
    ap.add_argument("--video-dir", default="data/ms_videos")
    ap.add_argument("--index-dir", default="runs/ms_index")
    ap.add_argument("--encoder", default="siglip2-base-224")
    ap.add_argument("--out", default="runs/temporal_why.json")
    args = ap.parse_args()

    queries = [q for q in load_queries(args.json, args.video_dir) if q.video_path]
    have = {os.path.splitext(f)[0] for f in os.listdir(args.index_dir) if f.endswith(".npz")}
    queries = [q for q in queries if q.video_id in have]
    print(f"{len(queries)} queries")

    from framesieve.encoders import CLIP_MODELS, ClipEncoder, SiglipEncoder
    enc = (ClipEncoder if args.encoder in CLIP_MODELS
           else SiglipEncoder)(args.encoder)
    qemb = np.concatenate(
        [enc.encode_text([q.text for q in queries[i:i + 256]]).cpu().numpy()
         for i in range(0, len(queries), 256)]).astype(np.float32)
    del enc

    cache: dict = {}
    out = {"chunk_sweep": [], "fps_sweep": []}

    print("\nvarying chunk length at 1 fps -- frames per chunk changes, seconds per frame does not")
    print(f"  {'chunk':>7}{'fr/chunk':>10}" + "".join(f"{'k='+str(k):>9}" for k in (1, 2, 3, 4, 6, 8, 12)))
    for cs in (5.0, 10.0, 20.0, 40.0):
        r = sweep(queries, qemb, args.index_dir, cs, (1, 2, 3, 4, 6, 8, 12), cache=cache)
        out["chunk_sweep"].append(r)
        best = max(r["by_k"], key=lambda k: r["by_k"][k]["R1"])
        print(f"  {cs:>6.0f}s{r['frames_per_chunk']:>10.1f}" +
              "".join(f"{r['by_k'][str(k)]['R1']:>9.2f}" for k in (1, 2, 3, 4, 6, 8, 12)) +
              f"   best k={best}")

    print("\nvarying frame rate at a fixed 10 s chunk -- frames per chunk changes too")
    print(f"  {'fps':>7}{'fr/chunk':>10}" + "".join(f"{'k='+str(k):>9}" for k in (1, 2, 3, 4, 6, 8)))
    for div, fps in ((4, 0.25), (2, 0.5), (1, 1.0)):
        r = sweep(queries, qemb, args.index_dir, 10.0, (1, 2, 3, 4, 6, 8),
                  fps_div=div, cache=cache)
        r["fps"] = fps
        out["fps_sweep"].append(r)
        best = max(r["by_k"], key=lambda k: r["by_k"][k]["R1"])
        print(f"  {fps:>7.2f}{r['frames_per_chunk']:>10.1f}" +
              "".join(f"{r['by_k'][str(k)]['R1']:>9.2f}" for k in (1, 2, 3, 4, 6, 8)) +
              f"   best k={best}")

    print("\nreading of the two sweeps:")
    for lab, key in (("chunk length", "chunk_sweep"), ("frame rate", "fps_sweep")):
        bests = [(r["frames_per_chunk"],
                  int(max(r["by_k"], key=lambda k: r["by_k"][k]["R1"])))
                 for r in out[key]]
        frac = [k / n for n, k in bests]
        print(f"  {lab:<14} best k: {[k for _, k in bests]}   "
              f"as a fraction of frames/chunk: {[round(f, 2) for f in frac]}")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
