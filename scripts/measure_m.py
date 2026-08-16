"""Does k* track m on real data, where m can be measured rather than assumed?

The synthetic in how_many_match.py predicts that the best k for mean-of-top-k
pooling tracks m, the number of items in a candidate that genuinely match. On
MomentSeeker m is unknown -- a chunk overlaps a labelled interval, but a label
says nothing about which frames actually carry the evidence.

Here it is known. The 4.5-hour video has a dense VLM pass over every one of its
16,244 frames for eight queries, so for any chunk and any query the number of
frames the oracle scored positive is a measurement, not an assumption.

The test: group chunks by their measured m, and for each group find the k that
best separates them from the negatives. If the synthetic is right, k* rises with
m and the endpoints lose exactly where it predicts they lose.

This also settles a loose end. On MomentSeeker the best k was 4 out of 10 frames
while the labelled moments cover ~8 of them, which is either a failure of the
account or evidence that a labelled interval is wider than the evidence inside
it. With a per-frame oracle, that is checkable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from framesieve.index import FrameIndex  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _indexio import read_index  # noqa: E402

CHUNK_S = 10.0


def load_groundtruth(path: str):
    """(frame_ts, scores[n_frames, n_queries], question strings)."""
    d = np.load(path, allow_pickle=True)
    return (d["ts"].astype(np.float64), d["scores"].astype(np.float64),
            [str(q) for q in d["queries"]])


def chunk_bounds(ts: np.ndarray, chunk_s: float):
    n = int(np.ceil((ts[-1] + 1.0) / chunk_s))
    ci = np.clip((ts / chunk_s).astype(int), 0, n - 1)
    order = np.argsort(ci, kind="stable")
    return n, ci, order, np.searchsorted(ci[order], np.arange(n + 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="runs/index_glasgow_siglip2b224.npz")
    ap.add_argument("--gt", default="runs/groundtruth_glasgow.npz")
    ap.add_argument("--queries", default="configs/queries_glasgow.json",
                    help="caption forms; the cheap encoder is a caption model "
                         "and the ground truth's queries are yes/no questions")
    ap.add_argument("--encoder", default="siglip2-base-224")
    ap.add_argument("--threshold", type=float, default=2.0,
                    help="oracle logit margin above which a frame counts as a match")
    ap.add_argument("--chunk-s", type=float, default=CHUNK_S)
    ap.add_argument("--out", default="runs/measure_m.json")
    args = ap.parse_args()

    g_ts, g_scores, questions = load_groundtruth(args.gt)
    idx = read_index(args.index)
    keep = idx.ts <= g_ts[-1] + 1e-6
    idx = FrameIndex(idx.ts[keep], idx.emb[keep], idx.seg_id[keep], idx.stats)
    E = idx.emb.astype(np.float32)
    print(f"{len(questions)} queries, {len(idx.ts):,} frames covered by the dense "
          f"pass, {args.chunk_s:.0f} s chunks, oracle threshold {args.threshold}")

    # The cheap encoder is caption-trained and the ground truth's queries are
    # yes/no questions written for the VLM; feeding the question form to SigLIP
    # costs real accuracy, so each stage sees the form it was built for.
    caps = list(questions)
    if os.path.exists(args.queries):
        by_q = {s["question"]: s["caption"]
                for s in json.load(open(args.queries))["queries"]}
        caps = [by_q.get(q, q) for q in questions]

    from framesieve.encoders import SiglipEncoder
    enc = SiglipEncoder(args.encoder)
    qemb = enc.encode_text(caps).cpu().numpy().astype(np.float32)
    del enc

    n_ch, ci, order, bounds = chunk_bounds(idx.ts, args.chunk_s)
    KS = (1, 2, 3, 4, 5, 6, 8, 10)

    # For every (query, chunk): the measured m, and the mean-of-top-k retrieval
    # score for each k. Chunks are pooled across queries and then grouped by m.
    recs = []
    for qi, cap in enumerate(caps):
        # align the oracle's frames onto the index's frames; both are 1 fps over
        # the same decode, so this is a check rather than a resampling
        pos = np.clip(np.searchsorted(g_ts, idx.ts), 0, len(g_ts) - 1)
        near = np.abs(g_ts[pos] - idx.ts) < 0.6
        oracle = np.where(near, g_scores[pos, qi], -np.inf)

        sims = E @ qemb[qi]
        s_ord, o_ord = sims[order], oracle[order]
        for c in range(n_ch):
            lo, hi = bounds[c], bounds[c + 1]
            if hi - lo < 3:
                continue
            v, ov = s_ord[lo:hi], o_ord[lo:hi]
            m = int(np.sum(np.isfinite(ov) & (ov >= args.threshold)))
            srt = np.sort(v)[::-1]
            recs.append(dict(q=qi, m=m,
                             sc={k: float(srt[:min(k, len(srt))].mean()) for k in KS}))

    groups = {"m=1": [1], "m=2": [2], "m=3-4": [3, 4], "m=5-7": [5, 6, 7],
              "m>=8": list(range(8, 100))}
    print(f"\nchunks by measured m (oracle score >= {args.threshold})")
    counts = {}
    for lab, ms in groups.items():
        counts[lab] = sum(1 for r in recs if r["m"] in ms)
    counts["m=0 (negatives)"] = sum(1 for r in recs if r["m"] == 0)
    for lab, n in counts.items():
        print(f"  {lab:<18}{n:>7}")

    # AUC of positives-in-this-group against the m=0 chunks of the same query,
    # per k. Comparing within a query keeps video-level score scale out of it.
    print("\nAUC against m=0 chunks of the same query, by k")
    print(f"  {'group':<12}" + "".join(f"{'k='+str(k):>8}" for k in KS)
          + f"{'best k':>9}")
    rows = []
    for lab, ms in groups.items():
        aucs = {k: [] for k in KS}
        for qi in range(len(caps)):
            neg = [r for r in recs if r["q"] == qi and r["m"] == 0]
            pos = [r for r in recs if r["q"] == qi and r["m"] in ms]
            if len(pos) < 3 or len(neg) < 10:
                continue
            for k in KS:
                a = np.array([r["sc"][k] for r in pos])
                b = np.array([r["sc"][k] for r in neg])
                x = np.concatenate([a, b])
                rk = np.argsort(np.argsort(x)) + 1
                aucs[k].append((rk[:len(a)].sum() - len(a) * (len(a) + 1) / 2)
                               / (len(a) * len(b)))
        if not aucs[KS[0]]:
            continue
        mean = {k: float(np.mean(v)) for k, v in aucs.items()}
        bk = max(mean, key=lambda k: mean[k])
        rows.append(dict(group=lab, ms=ms, n=counts[lab], auc=mean, best_k=bk,
                         n_queries=len(aucs[KS[0]])))
        print(f"  {lab:<12}" + "".join(f"{mean[k]:>8.3f}" for k in KS)
              + f"{bk:>9}")

    if len(rows) >= 3:
        mid = [float(np.mean(r["ms"][:4])) for r in rows]
        bk = [r["best_k"] for r in rows]
        c = float(np.corrcoef(np.log(mid), np.log(bk))[0, 1]) if np.std(bk) > 0 else 0.0
        print(f"\n  measured m:  {[round(v,1) for v in mid]}")
        print(f"  best k:      {bk}")
        print(f"  correlation of log(best k) with log(m): {c:+.3f}")
        print("  " + ("k* tracks the MEASURED number of matching frames"
                      if c > 0.8 else
                      "k* does NOT track the measured m on real data"))

    print("\nwhat the endpoints cost, per group")
    print(f"  {'group':<12}{'max (k=1)':>11}{'mean (k=10)':>13}{'best':>8}"
          f"{'max loses':>11}{'mean loses':>12}")
    for r in rows:
        a = r["auc"]
        print(f"  {r['group']:<12}{a[1]:>11.3f}{a[10]:>13.3f}{a[r['best_k']]:>8.3f}"
              f"{a[r['best_k']]-a[1]:>11.3f}{a[r['best_k']]-a[10]:>12.3f}")

    # ------------------------------------------------------------------
    # AUC says k=1 everywhere. But AUC is an average over the whole ranking and
    # the mechanism measured on MomentSeeker -- max being fooled by high-spread
    # chunks -- only bites at the HEAD of the ranking, where a single inflated
    # distractor is enough to take the top slot. R@1 is a head metric; AUC is
    # not. So run the synthetic's exact protocol on this real data: pools of one
    # positive against 49 negatives drawn from the same query, scored by whether
    # the top-ranked chunk is the positive.
    print("\npool protocol -- 1 positive vs 49 negatives from the same query, R@1")
    print(f"  {'group':<12}" + "".join(f"{'k='+str(k):>8}" for k in KS)
          + f"{'best k':>9}{'max loses':>11}")
    rng = np.random.default_rng(0)
    POOLS = 20000
    pool_rows = []
    by_q_neg = {qi: [r for r in recs if r["q"] == qi and r["m"] == 0]
                for qi in range(len(caps))}
    for lab, ms in groups.items():
        pos_all = [r for r in recs if r["m"] in ms and by_q_neg[r["q"]]]
        if len(pos_all) < 5:
            continue
        wins = {k: 0 for k in KS}
        for _ in range(POOLS):
            p = pos_all[rng.integers(len(pos_all))]
            negs = by_q_neg[p["q"]]
            pick = rng.choice(len(negs), size=min(49, len(negs)), replace=False)
            cand = [p] + [negs[j] for j in pick]
            for k in KS:
                sc = np.array([c["sc"][k] for c in cand])
                wins[k] += int(np.argmax(sc) == 0)
        acc = {k: v / POOLS for k, v in wins.items()}
        bk = max(acc, key=lambda k: acc[k])
        pool_rows.append(dict(group=lab, ms=ms, acc=acc, best_k=bk,
                              n_pos=len(pos_all)))
        print(f"  {lab:<12}" + "".join(f"{100*acc[k]:>8.1f}" for k in KS)
              + f"{bk:>9}{100*(acc[bk]-acc[1]):>11.1f}")

    if len(pool_rows) >= 3:
        mid = [float(np.mean(r["ms"][:4])) for r in pool_rows]
        bk = [r["best_k"] for r in pool_rows]
        c = (float(np.corrcoef(np.log(mid), np.log(bk))[0, 1])
             if np.std(bk) > 0 else 0.0)
        print(f"\n  measured m:  {[round(v,1) for v in mid]}")
        print(f"  best k:      {bk}")
        print(f"  correlation of log(best k) with log(m): {c:+.3f}")
        print("  " + ("k* tracks the measured m under a HEAD metric, while the "
                      "same data\n  under AUC says k=1 -- the pooling choice is "
                      "a head-of-ranking effect"
                      if c > 0.8 else
                      "k* does not track m under a head metric either"))

    with open(args.out, "w") as f:
        json.dump(dict(threshold=args.threshold, chunk_s=args.chunk_s,
                       counts=counts, auc_rows=rows, pool_rows=pool_rows),
                  f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
