"""Can a per-frame encoder be given temporal sense for free?

The cheap stage embeds each frame in isolation. Within a ten-second candidate
chunk it produces ten independent vectors and then throws nine of them away with
a max. That is the obvious place to look for something the encoder cannot see but
the *sequence* can.

Everything here operates on embeddings that are already cached, so each variant
costs a few seconds of numpy over 1,000 queries and nothing on the GPU. That is
the point: if temporal information is recoverable at query time, it is free.

Variants, in rough order of ambition:

  agg          how the frames in a chunk become one score: max (current), mean,
               top-2 mean, softmax-weighted mean
  smooth       the per-frame similarity sequence is noisy while real events
               persist for seconds, so convolve the score sequence with a box or
               triangular window before chunking -- a matched filter for "an
               event of about this length"
  delta        score the *change* between adjacent frames as well as the frames
               themselves. Motion is invisible to a still-frame encoder, but the
               rate of change of its embedding is a cheap proxy for it.
  prf          pseudo-relevance feedback: take the top-scoring frames, average
               their embeddings into the query, and re-rank. Standard in text
               retrieval, free here, and never applied to the query the user gave.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from framesieve.benchmarks.momentseeker import (  # noqa: E402
    chunks_for,
    gt_chunk_mask,
    load_queries,
    map_at_5_matched,
    recall_at_k,
)
from framesieve.index import FrameIndex  # noqa: E402

# --------------------------------------------------------------------------
# scoring variants
# --------------------------------------------------------------------------


def smooth(x: np.ndarray, w: int, kind: str = "box") -> np.ndarray:
    """Convolve a per-frame score sequence with a short window.

    A real event lasts several seconds, so its signature in the similarity
    sequence is a plateau rather than a spike; per-frame noise is a spike. A
    matched filter of roughly the event length should raise the former over the
    latter without costing anything.
    """
    # np.convolve(mode="same") returns max(len(x), w), so a window longer than
    # the sequence silently changes the array length. Short videos exist.
    w = min(int(w), len(x))
    if w <= 1:
        return x
    k = np.ones(w) if kind == "box" else np.bartlett(w + 2)[1:-1]
    k = k / k.sum()
    return np.convolve(x, k, mode="same")[: len(x)]


def chunk_scores(ts: np.ndarray, sims: np.ndarray, ch: np.ndarray,
                 agg: str = "max") -> np.ndarray:
    idx = np.clip(np.searchsorted(ch[:, 0], ts, "right") - 1, 0, len(ch) - 1)
    out = np.full(len(ch), -1e9, dtype=np.float64)
    for c in range(len(ch)):
        v = sims[idx == c]
        if v.size == 0:
            continue
        if agg == "max":
            out[c] = v.max()
        elif agg == "mean":
            out[c] = v.mean()
        elif agg == "top2":
            out[c] = np.sort(v)[-2:].mean()
        elif agg == "top3":
            out[c] = np.sort(v)[-3:].mean()
        elif agg == "softmax":
            p = np.exp((v - v.max()) * 12.0)
            out[c] = float((p * v).sum() / p.sum())
        else:
            raise ValueError(agg)
    return out


def frame_sims(idx: FrameIndex, q: np.ndarray, *, delta_w: float = 0.0,
               prf_k: int = 0, prf_w: float = 0.0) -> np.ndarray:
    """Per-frame similarity, optionally with motion and feedback terms."""
    E = idx.emb.astype(np.float32)
    s = E @ q

    if prf_k > 0 and prf_w > 0:
        # pseudo-relevance feedback: pull the query toward what it already
        # matched. Costs one extra matmul and no model call.
        top = np.argsort(-s)[:prf_k]
        qb = E[top].mean(0)
        qb /= np.linalg.norm(qb) + 1e-8
        q2 = (1 - prf_w) * q + prf_w * qb
        q2 /= np.linalg.norm(q2) + 1e-8
        s = E @ q2

    if delta_w > 0:
        # how fast the picture is changing, as a stand-in for motion the
        # still-frame encoder cannot represent
        d = np.zeros(len(E), dtype=np.float32)
        d[1:] = 1.0 - np.einsum("ij,ij->i", E[:-1], E[1:])
        d = (d - d.mean()) / (d.std() + 1e-8)
        s = s + delta_w * d * s.std()
    return s


# --------------------------------------------------------------------------


def evaluate(queries, qemb, index_dir, *, agg="max", smooth_w=1, smooth_kind="box",
             delta_w=0.0, prf_k=0, prf_w=0.0, cache=None) -> dict:
    cache = {} if cache is None else cache
    r1, r5, m5 = [], [], []
    for n, q in enumerate(queries):
        if q.video_id not in cache:
            if len(cache) > 40:
                cache.clear()
            cache[q.video_id] = FrameIndex.from_npz(
                os.path.join(index_dir, f"{q.video_id}.npz"))
        idx = cache[q.video_id]
        ch = chunks_for(float(idx.ts[-1]) + 1.0)
        s = frame_sims(idx, qemb[n], delta_w=delta_w, prf_k=prf_k, prf_w=prf_w)
        if smooth_w > 1:
            s = smooth(s, smooth_w, smooth_kind)
        cs = chunk_scores(idx.ts, s, ch, agg=agg)
        ranked = np.argsort(-cs)
        is_gt = gt_chunk_mask(ch, q.gt_intervals)
        r1.append(recall_at_k(ranked, is_gt, 1))
        r5.append(recall_at_k(ranked, is_gt, 5))
        m5.append(map_at_5_matched(ranked, ch, q.gt_intervals))
    return dict(R1=float(np.mean(r1)) * 100, R5=float(np.mean(r5)) * 100,
                mAP5=float(np.mean(m5)) * 100, n=len(r1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="data/ms_raw/t2v.json")
    ap.add_argument("--video-dir", default="data/ms_videos")
    ap.add_argument("--index-dir", default="runs/ms_index")
    ap.add_argument("--encoder", default="siglip2-base-224")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="runs/temporal.json")
    args = ap.parse_args()

    queries = [q for q in load_queries(args.json, args.video_dir) if q.video_path]
    have = {os.path.splitext(f)[0] for f in os.listdir(args.index_dir)
            if f.endswith(".npz")}
    queries = [q for q in queries if q.video_id in have]
    if args.limit:
        queries = queries[: args.limit]
    print(f"{len(queries)} queries")

    from framesieve.encoders import SiglipEncoder
    enc = SiglipEncoder(args.encoder)
    qemb = np.concatenate(
        [enc.encode_text([q.text for q in queries[i:i + 256]]).cpu().numpy()
         for i in range(0, len(queries), 256)]).astype(np.float32)
    del enc

    cache: dict = {}
    rows = []

    def run(label: str, **kw):
        t0 = time.perf_counter()
        r = evaluate(queries, qemb, args.index_dir, cache=cache, **kw)
        r.update(label=label, wall_s=time.perf_counter() - t0, **kw)
        rows.append(r)
        print(f"  {label:<34}R@1 {r['R1']:>6.2f}   R@5 {r['R5']:>6.2f}   "
              f"mAP@5 {r['mAP5']:>6.2f}", flush=True)
        return r

    print("\n--- how the frames in a chunk become one score ---")
    base = run("max (current)", agg="max")
    for a in ("mean", "top2", "top3", "softmax"):
        run(a, agg=a)

    print("\n--- smoothing the per-frame score sequence ---")
    for w in (3, 5, 9, 15, 25):
        run(f"box smooth w={w}", agg="max", smooth_w=w)
    for w in (5, 9, 15):
        run(f"triangular smooth w={w}", agg="max", smooth_w=w, smooth_kind="tri")

    print("\n--- change between adjacent frames as a motion proxy ---")
    for d in (0.1, 0.25, 0.5):
        run(f"delta weight={d}", agg="max", delta_w=d)

    print("\n--- pseudo-relevance feedback on the query ---")
    for k, w in ((5, 0.2), (5, 0.4), (20, 0.2), (20, 0.4), (50, 0.3)):
        run(f"prf k={k} w={w}", agg="max", prf_k=k, prf_w=w)

    print("\n--- best combinations ---")
    best_smooth = max((r for r in rows if "smooth" in r["label"]),
                      key=lambda r: r["R1"])
    best_prf = max((r for r in rows if r["label"].startswith("prf")),
                   key=lambda r: r["R1"])
    run("best smooth + best prf", agg="max",
        smooth_w=best_smooth.get("smooth_w", 1),
        smooth_kind=best_smooth.get("smooth_kind", "box"),
        prf_k=best_prf.get("prf_k", 0), prf_w=best_prf.get("prf_w", 0.0))
    run("best smooth + top2", agg="top2",
        smooth_w=best_smooth.get("smooth_w", 1),
        smooth_kind=best_smooth.get("smooth_kind", "box"))

    best = max(rows, key=lambda r: r["R1"])
    print(f"\nbaseline  R@1 {base['R1']:.2f}  mAP@5 {base['mAP5']:.2f}")
    print(f"best      R@1 {best['R1']:.2f}  mAP@5 {best['mAP5']:.2f}"
          f"   ({best['label']})")
    print(f"delta     R@1 {best['R1']-base['R1']:+.2f}  "
          f"mAP@5 {best['mAP5']-base['mAP5']:+.2f}")

    with open(args.out, "w") as f:
        json.dump(dict(config=vars(args), baseline=base, best=best, rows=rows),
                  f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
