"""What actually sets the optimal k? Third attempt.

The story so far, each version killed by the next:

  1. persistence   "a real event lasts several seconds, so require k consecutive
                   frames to agree." Killed by temporal_why.py: the best k tracks
                   the frames per chunk, not a number of seconds.

  2. order bias    "max over n samples is biased upward by ~1.5σ, so it rewards
                   high-variance chunks; debias it." Killed by estimator.py: the
                   debiasing this literally prescribes, max - λ·std at λ≈1.5, is
                   *worse* than max (-1.5 R@1). Meanwhile mean + λ·std, the
                   opposite sign, helps. Variance is informative, not a bias.

What every statistic that *does* work has in common -- top-k mean, power mean,
adaptive k, mean+λ·std -- is that it asks a chunk to be good over a *fraction* of
itself rather than at a single instant. So the third hypothesis is about the
labels rather than the model:

  3. coverage      a chunk counts as a positive when its IoU with a ground-truth
                   moment clears 0.3, so a positive chunk is one a fair fraction
                   of which is on target. The best k should be the number of
                   frames that fraction corresponds to -- and it should MOVE when
                   the IoU threshold moves.

That last clause is what makes it a real prediction rather than a restatement.
The threshold is a property of the benchmark's scoring, nothing to do with the
encoder or with time, so nothing else in the pipeline knows about it. If the
optimal k follows it, hypothesis 3 survives; if k sits still, it does not.
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
    iou,
    load_queries,
    recall_at_k,
)
from framesieve.index import FrameIndex  # noqa: E402


def chunks_of(duration_s: float, chunk_s: float) -> np.ndarray:
    n = max(1, int(np.ceil(duration_s / chunk_s)))
    st = np.arange(n, dtype=np.float64) * chunk_s
    return np.stack([st, np.minimum(st + chunk_s, duration_s)], axis=1)


def covered_fraction(chunks: np.ndarray, gts, mask: np.ndarray) -> list[float]:
    """For each positive chunk, how much of it is actually inside a GT moment."""
    out = []
    for c in np.flatnonzero(mask):
        a, b = chunks[c]
        cov = max((max(0.0, min(b, g[1]) - max(a, g[0])) for g in gts), default=0.0)
        if b > a:
            out.append(cov / (b - a))
    return out


def topk_scores(sims: np.ndarray, order, bounds, n_ch: int, k: int) -> np.ndarray:
    s = sims[order]
    out = np.full(n_ch, -1e9, dtype=np.float64)
    for c in range(n_ch):
        v = s[bounds[c]:bounds[c + 1]]
        if v.size:
            out[c] = np.sort(v)[-min(k, v.size):].mean()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="data/ms_raw/t2v.json")
    ap.add_argument("--video-dir", default="data/ms_videos")
    ap.add_argument("--index-dir", default="runs/ms_index")
    ap.add_argument("--encoder", default="siglip2-base-224")
    ap.add_argument("--chunk-s", type=float, default=10.0)
    ap.add_argument("--out", default="runs/why_k.json")
    args = ap.parse_args()

    queries = [q for q in load_queries(args.json, args.video_dir) if q.video_path]
    have = {os.path.splitext(f)[0] for f in os.listdir(args.index_dir)
            if f.endswith(".npz")}
    queries = [q for q in queries if q.video_id in have]
    print(f"{len(queries)} queries, {args.chunk_s:.0f} s chunks\n")

    from framesieve.encoders import SiglipEncoder
    enc = SiglipEncoder(args.encoder)
    qemb = np.concatenate(
        [enc.encode_text([q.text for q in queries[i:i + 256]]).cpu().numpy()
         for i in range(0, len(queries), 256)]).astype(np.float32)
    del enc

    KS = (1, 2, 3, 4, 5, 6, 8, 10)
    THRS = (0.1, 0.2, 0.3, 0.5, 0.7)

    # one pass over the videos; every threshold and every k scored from it
    hits = {(t, k): [] for t in THRS for k in KS}
    frac = {t: [] for t in THRS}
    n_pos = {t: 0 for t in THRS}
    cache: dict = {}
    for n, q in enumerate(queries):
        if q.video_id not in cache:
            if len(cache) > 40:
                cache.clear()
            cache[q.video_id] = FrameIndex.from_npz(
                os.path.join(args.index_dir, f"{q.video_id}.npz"))
        idx = cache[q.video_id]
        ch = chunks_of(float(idx.ts[-1]) + 1.0, args.chunk_s)
        sims = idx.emb.astype(np.float32) @ qemb[n]
        ci = np.clip(np.searchsorted(ch[:, 0], idx.ts, "right") - 1, 0, len(ch) - 1)
        order = np.argsort(ci, kind="stable")
        bounds = np.searchsorted(ci[order], np.arange(len(ch) + 1))

        ranked = {k: np.argsort(-topk_scores(sims, order, bounds, len(ch), k))
                  for k in KS}
        for t in THRS:
            m = gt_chunk_mask(ch, q.gt_intervals, thr=t)
            n_pos[t] += int(m.sum())
            frac[t].extend(covered_fraction(ch, q.gt_intervals, m))
            for k in KS:
                hits[(t, k)].append(recall_at_k(ranked[k], m, 1))

    print("how much of a positive chunk is actually on target, by IoU threshold")
    print(f"  {'IoU thr':>8}{'pos chunks':>12}{'median cov':>12}{'mean cov':>10}")
    for t in THRS:
        f = np.array(frac[t]) if frac[t] else np.array([np.nan])
        print(f"  {t:>8.1f}{n_pos[t]:>12}{np.median(f):>12.2f}{f.mean():>10.2f}")

    print("\nR@1 by k, at each IoU threshold  (the prediction: best k rises with thr)")
    print(f"  {'IoU thr':>8}" + "".join(f"{'k='+str(k):>8}" for k in KS)
          + f"{'best k':>9}{'k/n':>7}{'coverage':>10}")
    rows = []
    for t in THRS:
        r = {k: float(np.mean(hits[(t, k)])) * 100 for k in KS}
        bk = max(r, key=lambda k: r[k])
        cov = float(np.median(frac[t])) if frac[t] else float("nan")
        rows.append(dict(thr=t, by_k=r, best_k=bk,
                         k_over_n=bk / args.chunk_s, coverage=cov))
        print(f"  {t:>8.1f}" + "".join(f"{r[k]:>8.2f}" for k in KS)
              + f"{bk:>9}{bk/args.chunk_s:>7.2f}{cov:>10.2f}")

    ks = np.array([r["best_k"] for r in rows], float)
    cs = np.array([r["coverage"] for r in rows], float)
    ok = np.isfinite(cs)
    print("\nverdict")
    if ok.sum() > 2 and np.std(ks[ok]) > 0:
        c = float(np.corrcoef(ks[ok], cs[ok])[0, 1])
        print(f"  best k across thresholds: {[int(k) for k in ks]}")
        print(f"  median coverage:          {[round(float(v), 2) for v in cs]}")
        print(f"  correlation: {c:+.2f}")
        print("  hypothesis 3 SURVIVES -- optimal k follows the label threshold"
              if c > 0.5 else
              "  hypothesis 3 FAILS -- optimal k does not follow the label threshold")
    else:
        print(f"  best k is {[int(k) for k in ks]} at every threshold: "
              "flat, so the label threshold does not set it. Hypothesis 3 FAILS.")

    with open(args.out, "w") as f:
        json.dump(dict(chunk_s=args.chunk_s, rows=rows,
                       coverage={str(t): frac[t][:2000] for t in THRS}), f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
