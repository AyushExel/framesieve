"""Why an averaged metric cannot see a pooling change.

Three separate domains in this project produced the same odd split:

  real video, dense oracle   R@1 says the best k tracks m (correlation +0.94);
                             AUC on the identical scores says k=1 everywhere
                             (+0.37).
  MomentSeeker               the win shows up in R@1 and mAP@5.
  single-vector RAG          the head metric picks a larger k than nDCG@10 in
                             four of six configurations.

That is a claim about EVALUATION rather than about pooling, and it is the more
broadly useful of the two: it says a whole class of improvement is invisible to
the metric most teams tune against. It deserves a decisive test rather than three
suggestive ones.

Here m is fixed, so the only thing varying is the metric. If the account is
right -- max's failure is that a single inflated negative takes the TOP slot,
and nothing worse than that -- then the measured optimum should slide toward
k = 1 as the metric looks further down the ranking, and sit furthest from 1 at
R@1. If the optimum is the same under every metric, the split seen in the real
data was a coincidence and should be reported as one.
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def pool(rng, n, m, mu, n_pos, n_neg, tail_frac, tail_sigma):
    """One ranking pool.

    The negatives are a MIXTURE, and that detail is the whole experiment. Giving
    every negative extra within-candidate spread just penalises max uniformly and
    moves the optimum to the mean for every metric -- measured, and it produced a
    degenerate table. What the real data actually showed is a TAIL: most false
    chunks are ordinary, and a minority (the ones straddling a cut, measured at
    +0.65 SD of score spread) are volatile enough for their maximum to reach the
    top of the ranking.

    A rare high-spread distractor can only ever displace a few items at the very
    head. That is why the metric should matter: a head metric pays the full cost
    of it, and a metric averaging over a hundred candidates barely notices.
    """
    pos = rng.normal(0.0, 1.0, size=(n_pos, n))
    pos[:, :m] = rng.normal(mu, 1.0, size=(n_pos, m))
    sig = np.where(rng.random(n_neg) < tail_frac, tail_sigma, 1.0)[:, None]
    neg = rng.normal(0.0, 1.0, size=(n_neg, n)) * sig
    x = np.concatenate([pos, neg])
    is_pos = np.zeros(len(x), bool)
    is_pos[:n_pos] = True
    return x, is_pos


def metrics(scores, is_pos, ndcg_k=10) -> dict:
    """Every metric from one ranking, so they cannot disagree for any reason
    other than what they measure."""
    order = np.argsort(-scores)
    rel = is_pos[order]
    n_pos = int(is_pos.sum())
    first = int(np.argmax(rel)) if rel.any() else len(rel)
    disc = 1.0 / np.log2(np.arange(2, ndcg_k + 2))
    dcg = float((rel[:ndcg_k] * disc).sum())
    idcg = float(disc[:min(n_pos, ndcg_k)].sum())
    ranks = np.flatnonzero(rel) + 1
    return {
        "R@1": float(rel[0]),
        "R@5": float(rel[:5].any()),
        "R@20": float(rel[:20].any()),
        "MRR": 1.0 / (first + 1),
        "nDCG@10": dcg / idcg if idcg > 0 else 0.0,
        # mean reciprocal over ALL positives, which weights the deep ones too
        "MAP": float(np.mean([(i + 1) / r for i, r in enumerate(ranks)]))
        if ranks.size else 0.0,
        # AUC: probability a random positive outranks a random negative
        "AUC": float(((len(rel) - np.flatnonzero(rel)) - np.arange(n_pos, 0, -1)
                      ).sum() / (n_pos * (len(rel) - n_pos)))
        if 0 < n_pos < len(rel) else 0.5,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32, help="items per candidate")
    ap.add_argument("--m", type=int, default=8, help="items that genuinely match")
    ap.add_argument("--mu", type=float, default=1.6)
    ap.add_argument("--n-pos", type=int, default=5)
    ap.add_argument("--n-neg", type=int, default=95)
    ap.add_argument("--tail-frac", type=float, default=0.05,
                    help="share of negatives that are volatile distractors")
    ap.add_argument("--tail-sigma", type=float, default=2.0,
                    help="their within-candidate spread, against 1.0 for the rest")
    ap.add_argument("--pools", type=int, default=6000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="runs/metric_hides_it.json")
    args = ap.parse_args()

    KS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
    NAMES = ["R@1", "R@5", "R@20", "MRR", "nDCG@10", "MAP", "AUC"]

    print(f"n={args.n}, m={args.m} FIXED, {args.n_pos} positives vs "
          f"{args.n_neg} negatives, of which {100*args.tail_frac:.0f}% are "
          f"distractors at sigma={args.tail_sigma}")
    print(f"{args.pools} pools x {args.seeds} seeds. Only the metric varies.\n")

    acc = {name: {k: [] for k in KS} for name in NAMES}
    for s in range(args.seeds):
        rng = np.random.default_rng(7000 + s)
        run = {name: {k: 0.0 for k in KS} for name in NAMES}
        for _ in range(args.pools):
            x, is_pos = pool(rng, args.n, args.m, args.mu, args.n_pos,
                             args.n_neg, args.tail_frac, args.tail_sigma)
            srt = np.sort(x, axis=1)[:, ::-1]
            for k in KS:
                sc = srt[:, :min(k, args.n)].mean(axis=1)
                for name, v in metrics(sc, is_pos).items():
                    run[name][k] += v
        for name in NAMES:
            for k in KS:
                acc[name][k].append(run[name][k] / args.pools)

    print(f"  {'metric':<10}" + "".join(f"{'k='+str(k):>8}" for k in KS)
          + f"{'best k':>9}{'gain over max':>15}")
    rows = []
    for name in NAMES:
        mean = {k: float(np.mean(acc[name][k])) for k in KS}
        bk = max(mean, key=lambda k: mean[k])
        # per-seed best k, so a flat curve cannot masquerade as a clean answer
        per_seed = [KS[int(np.argmax([acc[name][k][i] for k in KS]))]
                    for i in range(args.seeds)]
        gain = 100 * (mean[bk] - mean[1]) / max(mean[1], 1e-9)
        rows.append(dict(metric=name, by_k=mean, best_k=bk,
                         best_k_per_seed=per_seed, rel_gain_pct=gain))
        print(f"  {name:<10}" + "".join(f"{mean[k]:>8.3f}" for k in KS)
              + f"{bk:>9}{gain:>14.1f}%")

    print(f"\n  {'metric':<10}{'best k':>9}{'per seed':>28}")
    for r in rows:
        print(f"  {r['metric']:<10}{r['best_k']:>9}{str(r['best_k_per_seed']):>28}")

    depth = {"R@1": 1, "R@5": 5, "MRR": 10, "nDCG@10": 10, "R@20": 20,
             "MAP": 100, "AUC": 100}
    d = np.array([depth[r["metric"]] for r in rows], float)
    b = np.array([r["best_k"] for r in rows], float)
    c = float(np.corrcoef(np.log(d), np.log(b))[0, 1]) if np.std(b) > 0 else 0.0
    print(f"\n  correlation of log(best k) with log(how deep the metric looks): "
          f"{c:+.3f}")
    head, deep = rows[0]["best_k"], rows[-1]["best_k"]
    print(f"  R@1 picks k={head}; AUC picks k={deep}")
    print("  " + ("the deeper the metric looks, the closer to max it says to sit "
                  "-- so\n  an averaged metric will tell you this dial does not "
                  "exist" if c < -0.5 else
                  "the metric does NOT move the optimum -- the split seen in the "
                  "real data\n  was not caused by metric depth"))

    with open(args.out, "w") as f:
        json.dump(dict(config=vars(args), ks=list(KS), rows=rows,
                       corr_depth=c), f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
