"""Take the estimator-bias explanation seriously and see how far it goes.

scripts/temporal_why.py established that replacing `max` with the mean of the top
k frames in a chunk is an order-statistic effect, not a temporal one: the best k
tracks the number of frames per chunk, not a number of seconds. If that reading is
right it makes three further predictions, and they are all testable in numpy over
embeddings that already exist.

  power mean    max and mean are the two ends of one family,
                (1/n Σ s_i^p)^(1/p): p=1 is the mean, p→∞ is the max. If top-k is
                just "a less extreme statistic", some p in between should do at
                least as well, and the family is smoother to tune.

  variance      the bias in max over n samples grows with the spread of those
                samples, so chunks that merely *vary* a lot -- ones spanning a
                cut, say -- outrank chunks that actually contain the answer.
                Subtracting λ·std should therefore recover much of the top-k gain
                on its own. If it does not, the explanation is incomplete.

  adaptive k    if the best k is a fraction of n rather than a constant, then
                k = round(n/f) should beat any fixed k once chunk lengths vary --
                and, more usefully, should need no retuning when they do.

The third is the only one that is a recipe rather than a diagnosis. A fixed k=4 is
tuned to this benchmark's 10-second chunks; k = n/f is tuned to nothing.
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


def grouped(ts: np.ndarray, ch: np.ndarray):
    """Frame indices sorted by chunk, with the boundary offsets into them."""
    idx = np.clip(np.searchsorted(ch[:, 0], ts, "right") - 1, 0, len(ch) - 1)
    order = np.argsort(idx, kind="stable")
    return order, np.searchsorted(idx[order], np.arange(len(ch) + 1))


def aggregate(sims: np.ndarray, order, bounds, n_ch: int, mode: str,
              param: float) -> np.ndarray:
    """One chunk score per chunk, by the named statistic."""
    s = sims[order]
    out = np.full(n_ch, -1e9, dtype=np.float64)
    for c in range(n_ch):
        v = s[bounds[c]:bounds[c + 1]]
        if v.size == 0:
            continue
        if mode == "max":
            out[c] = v.max()
        elif mode == "topk":
            out[c] = np.sort(v)[-min(int(param), v.size):].mean()
        elif mode == "adaptive":
            # k as a fraction of the frames present, which is what temporal_why
            # says the optimum actually tracks
            k = max(1, int(round(v.size / param)))
            out[c] = np.sort(v)[-min(k, v.size):].mean()
        elif mode == "power":
            # similarities are cosine and can be negative, so shift to positive
            # before exponentiating; the shift is constant across chunks in a
            # video and so cannot change their order on its own
            w = v - sims.min() + 1e-6
            out[c] = float(np.mean(w ** param) ** (1.0 / param))
        elif mode == "varpen":
            out[c] = v.max() - param * (v.std() if v.size > 1 else 0.0)
        elif mode == "varpen_mean":
            out[c] = v.mean() + param * (v.std() if v.size > 1 else 0.0)
        else:
            raise ValueError(mode)
    return out


def evaluate(queries, qemb, index_dir, chunk_s, variants, cache=None) -> dict:
    """Score every variant in one pass over the videos -- loading the index is
    the expensive part, the statistics are free."""
    cache = {} if cache is None else cache
    acc = {v: ([], []) for v in variants}
    for n, q in enumerate(queries):
        if q.video_id not in cache:
            if len(cache) > 40:
                cache.clear()
            cache[q.video_id] = FrameIndex.load(
                os.path.join(index_dir, f"{q.video_id}.npz"))
        idx = cache[q.video_id]
        ch = chunks_of(float(idx.ts[-1]) + 1.0, chunk_s)
        sims = idx.emb.astype(np.float32) @ qemb[n]
        order, bounds = grouped(idx.ts, ch)
        is_gt = gt_chunk_mask(ch, q.gt_intervals)
        for v in variants:
            cs = aggregate(sims, order, bounds, len(ch), v[0], v[1])
            ranked = np.argsort(-cs)
            acc[v][0].append(recall_at_k(ranked, is_gt, 1))
            acc[v][1].append(map_at_5_matched(ranked, ch, q.gt_intervals))
    return {v: dict(R1=float(np.mean(a)) * 100, mAP5=float(np.mean(b)) * 100)
            for v, (a, b) in acc.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="data/ms_raw/t2v.json")
    ap.add_argument("--video-dir", default="data/ms_videos")
    ap.add_argument("--index-dir", default="runs/ms_index")
    ap.add_argument("--encoder", default="siglip2-base-224")
    ap.add_argument("--out", default="runs/estimator.json")
    args = ap.parse_args()

    queries = [q for q in load_queries(args.json, args.video_dir) if q.video_path]
    have = {os.path.splitext(f)[0] for f in os.listdir(args.index_dir)
            if f.endswith(".npz")}
    queries = [q for q in queries if q.video_id in have]
    print(f"{len(queries)} queries")

    from framesieve.encoders import CLIP_MODELS, ClipEncoder, SiglipEncoder
    enc = (ClipEncoder if args.encoder in CLIP_MODELS
           else SiglipEncoder)(args.encoder)
    qemb = np.concatenate(
        [enc.encode_text([q.text for q in queries[i:i + 256]]).cpu().numpy()
         for i in range(0, len(queries), 256)]).astype(np.float32)
    del enc

    variants = ([("max", 0.0), ("topk", 4.0)]
                + [("power", p) for p in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)]
                + [("varpen", l) for l in (0.25, 0.5, 1.0, 1.5, 2.0)]
                + [("varpen_mean", l) for l in (0.5, 1.0, 1.5)]
                + [("adaptive", f) for f in (1.5, 2.0, 2.5, 3.0, 4.0)])

    out = {}
    cache: dict = {}
    for chunk_s in (10.0, 20.0, 40.0):
        print(f"\n--- chunk length {chunk_s:.0f} s "
              f"(~{chunk_s:.0f} frames per chunk at 1 fps) ---")
        r = evaluate(queries, qemb, args.index_dir, chunk_s, variants, cache=cache)
        base = r[("max", 0.0)]["R1"]
        out[str(chunk_s)] = {f"{m}:{p}": v for (m, p), v in r.items()}
        for (m, p), v in r.items():
            lab = m if m == "max" else f"{m} {p:g}"
            print(f"  {lab:<20}R@1 {v['R1']:>6.2f} ({v['R1']-base:+5.2f})   "
                  f"mAP@5 {v['mAP5']:>6.2f}")
        best = max(r, key=lambda k: r[k]["R1"])
        print(f"  best: {best[0]} {best[1]:g}  R@1 {r[best]['R1']:.2f}")

    print("\ndoes one setting hold across chunk lengths?")
    for m in ("topk", "adaptive"):
        for p in sorted({p for mm, p in variants if mm == m}):
            row = [out[str(c)][f"{m}:{p}"]["R1"] for c in (10.0, 20.0, 40.0)]
            print(f"  {m} {p:g}:  " + "  ".join(f"{v:6.2f}" for v in row) +
                  f"   spread {max(row)-min(row):.2f}")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
