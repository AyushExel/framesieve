"""Stop hypothesising and look at which chunks change places.

Three explanations for the top-k gain have now been tested and killed
(persistence, order-statistic debiasing, label coverage -- see temporal_why.py,
estimator.py, why_k.py). Rather than invent a fourth, this asks the mechanical
question directly:

    when max and top-k disagree, WHICH chunks move, and what are they like?

R@1 changes only through the rank of the best true chunk, and that rank is just a
count of how many false chunks outscore it. So the gain decomposes exactly two
ways -- the true chunk scored higher, or fewer false chunks did -- and the false
chunks that get demoted can be described.

Two descriptors, both free from the cached embeddings:

    spread   the standard deviation of the frame scores inside the chunk. If max
             is being fooled by variance, demoted distractors should have more of
             it than the average false chunk.
    churn    the mean 1 - cos between adjacent frame embeddings in the chunk: how
             much the picture changes across it. A chunk straddling a shot cut has
             high churn, and is exactly the kind that can contain one unrepresentative
             frame that happens to match.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from framesieve.benchmarks.momentseeker import gt_chunk_mask, load_queries  # noqa: E402
from framesieve.index import FrameIndex  # noqa: E402


def chunks_of(duration_s: float, chunk_s: float) -> np.ndarray:
    n = max(1, int(np.ceil(duration_s / chunk_s)))
    st = np.arange(n, dtype=np.float64) * chunk_s
    return np.stack([st, np.minimum(st + chunk_s, duration_s)], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="data/ms_raw/t2v.json")
    ap.add_argument("--video-dir", default="data/ms_videos")
    ap.add_argument("--index-dir", default="runs/ms_index")
    ap.add_argument("--encoder", default="siglip2-base-224")
    ap.add_argument("--chunk-s", type=float, default=10.0)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out", default="runs/what_k_fixes.json")
    args = ap.parse_args()

    queries = [q for q in load_queries(args.json, args.video_dir) if q.video_path]
    have = {os.path.splitext(f)[0] for f in os.listdir(args.index_dir)
            if f.endswith(".npz")}
    queries = [q for q in queries if q.video_id in have]
    print(f"{len(queries)} queries, {args.chunk_s:.0f} s chunks, k={args.k}\n")

    from framesieve.encoders import SiglipEncoder
    enc = SiglipEncoder(args.encoder)
    qemb = np.concatenate(
        [enc.encode_text([q.text for q in queries[i:i + 256]]).cpu().numpy()
         for i in range(0, len(queries), 256)]).astype(np.float32)
    del enc

    # per-query: rank of the best true chunk under each statistic
    rank_max, rank_topk = [], []
    # descriptors, z-scored within each video so videos of different character
    # cannot dominate the pooled mean
    demoted = {"spread": [], "churn": []}      # false chunks max ranked above the
    allneg = {"spread": [], "churn": []}       # best true chunk, that top-k did not
    postrue = {"spread": [], "churn": []}      # true chunks, for contrast
    n_demoted = 0

    cache: dict = {}
    for n, q in enumerate(queries):
        if q.video_id not in cache:
            if len(cache) > 40:
                cache.clear()
            cache[q.video_id] = FrameIndex.from_npz(
                os.path.join(args.index_dir, f"{q.video_id}.npz"))
        idx = cache[q.video_id]
        E = idx.emb.astype(np.float32)
        ch = chunks_of(float(idx.ts[-1]) + 1.0, args.chunk_s)
        sims = E @ qemb[n]
        ci = np.clip(np.searchsorted(ch[:, 0], idx.ts, "right") - 1, 0, len(ch) - 1)
        order = np.argsort(ci, kind="stable")
        bounds = np.searchsorted(ci[order], np.arange(len(ch) + 1))
        s_ord, E_ord = sims[order], E[order]

        n_ch = len(ch)
        s_max = np.full(n_ch, -1e9)
        s_top = np.full(n_ch, -1e9)
        spread = np.zeros(n_ch)
        churn = np.zeros(n_ch)
        for c in range(n_ch):
            v = s_ord[bounds[c]:bounds[c + 1]]
            if v.size == 0:
                continue
            s_max[c] = v.max()
            s_top[c] = np.sort(v)[-min(args.k, v.size):].mean()
            spread[c] = v.std() if v.size > 1 else 0.0
            W = E_ord[bounds[c]:bounds[c + 1]]
            churn[c] = float(np.mean(1.0 - np.einsum("ij,ij->i", W[:-1], W[1:]))) \
                if W.shape[0] > 1 else 0.0

        is_gt = gt_chunk_mask(ch, q.gt_intervals)
        if not is_gt.any():
            continue

        # rank of the best true chunk = 1 + number of false chunks scoring above it
        def rank_of_best_true(sc):
            best = sc[is_gt].max()
            return int((sc[~is_gt] > best).sum())
        rm, rt = rank_of_best_true(s_max), rank_of_best_true(s_top)
        rank_max.append(rm)
        rank_topk.append(rt)

        # z-score the descriptors within this video
        def z(x):
            sd = x.std()
            return (x - x.mean()) / sd if sd > 1e-9 else np.zeros_like(x)
        zs, zc = z(spread), z(churn)

        above_max = (~is_gt) & (s_max > s_max[is_gt].max())
        above_top = (~is_gt) & (s_top > s_top[is_gt].max())
        dem = above_max & ~above_top
        n_demoted += int(dem.sum())
        if dem.any():
            demoted["spread"].extend(zs[dem].tolist())
            demoted["churn"].extend(zc[dem].tolist())
        neg = ~is_gt
        if neg.any():
            allneg["spread"].extend(zs[neg].tolist())
            allneg["churn"].extend(zc[neg].tolist())
        # true chunks matter to the argument: if they are ALSO high-spread, then
        # no blanket variance penalty can separate them from the distractors, and
        # that is exactly why max - lambda*std failed while top-k works
        postrue["spread"].extend(zs[is_gt].tolist())
        postrue["churn"].extend(zc[is_gt].tolist())

    rank_max = np.array(rank_max)
    rank_topk = np.array(rank_topk)
    print("rank of the best true chunk (0 = top of the list, so R@1 counts zeros)")
    print(f"  {'statistic':<12}{'mean':>9}{'median':>9}{'rank 0':>9}")
    for lab, r in (("max", rank_max), (f"top-{args.k}", rank_topk)):
        print(f"  {lab:<12}{r.mean():>9.1f}{np.median(r):>9.1f}"
              f"{100*np.mean(r == 0):>8.1f}%")

    better = int((rank_topk < rank_max).sum())
    worse = int((rank_topk > rank_max).sum())
    print(f"\n  top-{args.k} ranks the true chunk higher on {better} queries, "
          f"lower on {worse}, unchanged on {len(rank_max)-better-worse}")

    print(f"\nthe {n_demoted} false chunks top-{args.k} pushed back below the true one,")
    print("described against all false chunks (both z-scored within video)")
    print(f"  {'descriptor':<12}{'demoted':>10}{'all false':>12}{'difference':>13}"
          f"{'true chunks':>14}")
    res = {}
    for key in ("spread", "churn"):
        d = np.array(demoted[key]) if demoted[key] else np.array([np.nan])
        a = np.array(allneg[key]) if allneg[key] else np.array([np.nan])
        t = np.array(postrue[key]) if postrue[key] else np.array([np.nan])
        res[key] = dict(demoted=float(d.mean()), all=float(a.mean()),
                        true=float(t.mean()), n_demoted=int(d.size),
                        n_all=int(a.size), n_true=int(t.size))
        print(f"  {key:<12}{d.mean():>+10.2f}{a.mean():>+12.2f}"
              f"{d.mean()-a.mean():>+13.2f}{t.mean():>+14.2f}")

    print("\nwhy penalising variance outright does not work")
    for key in ("spread", "churn"):
        r = res[key]
        print(f"  {key}: distractors {r['demoted']:+.2f}, true chunks "
              f"{r['true']:+.2f} -- "
              + ("both elevated, so a blanket penalty removes signal with noise"
                 if r["true"] > 0.15 else
                 "true chunks are not elevated, so a blanket penalty should work"))

    print("\nreading")
    ds, dc = res["spread"], res["churn"]
    for key, r in (("spread", ds), ("churn", dc)):
        diff = r["demoted"] - r["all"]
        verdict = ("distinctly higher -- consistent with max being fooled by it"
                   if diff > 0.15 else
                   "distinctly lower" if diff < -0.15 else
                   "indistinguishable from an average false chunk")
        print(f"  {key:<8}{verdict}")

    with open(args.out, "w") as f:
        json.dump(dict(chunk_s=args.chunk_s, k=args.k,
                       rank_max=dict(mean=float(rank_max.mean()),
                                     r0=float(np.mean(rank_max == 0))),
                       rank_topk=dict(mean=float(rank_topk.mean()),
                                      r0=float(np.mean(rank_topk == 0))),
                       better=better, worse=worse, descriptors=res), f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
