"""One knob explains video chunks, ColBERT's inner max, and ColBERT's outer sum.

The measurements that need explaining, all from this project:

  video chunks     10 frames per candidate, `max` is wrong, the best statistic is
                   the mean of the top ~4, and four unrelated families of
                   statistic all peak there.
  ColBERT inner    ~300 document tokens per query token, `max` is right, and
                   moving into the interior hurts monotonically and severely
                   (-2.6 at k=2, -18.0 at k=16 on SciFact).
  ColBERT outer    ~32 query tokens per document, the sum (= mean) is the
                   incumbent, and moving toward the max helps on NFCorpus
                   (+3.9) and is neutral on SciFact.

Those look like three unrelated facts. They are one fact with three values of a
single quantity: HOW MANY of the n things being pooled are supposed to match.

  one     a query token matches one place in a document -> max
  several a moment occupies several frames of a chunk   -> the interior
  all     every query token contributes to a document's score -> mean

max and mean are not two different tools. They are the two endpoints of one
family, and the tuned parameter is the answer to "how many of them should
match?" -- a question about the data, which is why nobody thinks of it as a
hyperparameter and everybody sets it by accident.

This script tests that as a prediction rather than a story. Build a scoring
problem where the number of genuinely matching items m is KNOWN by construction,
sweep it, and check the claim:

    the optimal k should track m.

If the optimum sits at k=1 for every m, or drifts with n instead, the unifying
account is wrong and the three measurements above stay three unrelated facts.
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def trial(rng, n_pos, n_neg, n, m, mu, sigma_neg=1.0, ks=(1,)):
    """One retrieval pool.

    A positive candidate has m items drawn from N(mu, 1) -- the parts of it that
    genuinely match -- and n - m from N(0, 1). A negative candidate is all noise.
    sigma_neg > 1 gives the negatives extra within-candidate spread, which is the
    "chunk straddling a cut" case measured in the real data.

    Scored by mean-of-top-k; reported as the share of pools whose top-ranked
    candidate is a positive, which is R@1.
    """
    pos = rng.normal(0.0, 1.0, size=(n_pos, n))
    if m > 0:
        pos[:, :m] = rng.normal(mu, 1.0, size=(n_pos, m))
    neg = rng.normal(0.0, sigma_neg, size=(n_neg, n))
    allc = np.concatenate([pos, neg])
    is_pos = np.zeros(len(allc), bool)
    is_pos[:n_pos] = True
    srt = np.sort(allc, axis=1)[:, ::-1]
    out = {}
    for k in ks:
        sc = srt[:, :min(k, n)].mean(axis=1)
        out[k] = bool(is_pos[int(np.argmax(sc))])
    return out


def sweep(rng, ms, ks, n, mu, n_pos, n_neg, pools, sigma_neg=1.0):
    acc = {m: {k: 0 for k in ks} for m in ms}
    for m in ms:
        for _ in range(pools):
            r = trial(rng, n_pos, n_neg, n, m, mu, sigma_neg, ks)
            for k, hit in r.items():
                acc[m][k] += int(hit)
    return {m: {k: v / pools for k, v in d.items()} for m, d in acc.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32, help="items per candidate")
    ap.add_argument("--mu", type=float, default=1.6, help="signal strength")
    ap.add_argument("--pools", type=int, default=4000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="runs/how_many_match.json")
    args = ap.parse_args()

    ks = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
    ms = (1, 2, 4, 8, 16, 32)

    print(f"n = {args.n} items per candidate, signal mu = {args.mu}, "
          f"{args.pools} pools x {args.seeds} seeds\n")
    print("R@1 by k, for each number of genuinely matching items m")
    print(f"  {'m':>4}" + "".join(f"{'k='+str(k):>8}" for k in ks)
          + f"{'best k':>9}{'best k / m':>12}")

    per_seed = {m: [] for m in ms}
    tables = {}
    for m in ms:
        rows = []
        for s in range(args.seeds):
            rng = np.random.default_rng(1000 + s)
            r = sweep(rng, (m,), ks, args.n, args.mu, 1, 49, args.pools)[m]
            rows.append([r[k] for k in ks])
            per_seed[m].append(int(ks[int(np.argmax(rows[-1]))]))
        mean = np.mean(rows, axis=0)
        bk = ks[int(np.argmax(mean))]
        tables[m] = dict(ks=list(ks), r1=[float(v) for v in mean],
                         best_k=bk, best_k_per_seed=per_seed[m])
        print(f"  {m:>4}" + "".join(f"{100*v:>8.1f}" for v in mean)
              + f"{bk:>9}{bk/m:>12.2f}")

    # A row where every k reaches ~100% has no optimum to find: when all n items
    # carry signal the task is trivial and argmax over ties is arbitrary. Those
    # rows are excluded from the fit and marked, rather than quietly included.
    SAT = 0.98
    usable = [m for m in ms if max(tables[m]["r1"]) < SAT]
    best = [tables[m]["best_k"] for m in ms]
    print(f"\n  m       {list(ms)}")
    print(f"  best k  {best}"
          + ("   (excluding " + str([m for m in ms if m not in usable])
             + ", saturated at ~100% for every k)" if len(usable) < len(ms) else ""))
    if len(usable) >= 3:
        c = float(np.corrcoef(np.log(usable),
                              np.log([tables[m]["best_k"] for m in usable]))[0, 1])
        print(f"  correlation of log(best k) with log(m): {c:+.3f}  "
              f"(n = {len(usable)} unsaturated)")
        print("  " + ("the optimum tracks m, so 'how many should match' is the knob"
                      if c > 0.9 else
                      "the optimum does NOT track m -- the unifying account fails"))
    print("  per-seed best k, to show the optimum is not noise:")
    for m in ms:
        print(f"    m={m:<3} {tables[m]['best_k_per_seed']}")

    # the two endpoints people actually use, priced against the tuned interior
    print("\n  what each endpoint costs, against the best k for that m")
    print(f"  {'m':>4}{'max (k=1)':>12}{'mean (k=n)':>13}{'best k':>9}"
          f"{'best R@1':>11}{'max loses':>11}{'mean loses':>12}")
    endpoint_rows = []
    for m in ms:
        r = tables[m]["r1"]
        bi = int(np.argmax(r))
        e = dict(m=m, max=r[0], mean=r[-1], best_k=ks[bi], best=r[bi])
        endpoint_rows.append(e)
        print(f"  {m:>4}{100*r[0]:>12.1f}{100*r[-1]:>13.1f}{ks[bi]:>9}"
              f"{100*r[bi]:>11.1f}{100*(r[bi]-r[0]):>11.1f}"
              f"{100*(r[bi]-r[-1]):>12.1f}")

    # does giving the negatives extra spread punish max specifically? That is the
    # "distractor straddling a cut" case measured on real video.
    print("\n  with high-variance negatives (the distractor case), m = 4")
    print(f"  {'sigma_neg':>10}{'max (k=1)':>12}{'best k':>9}{'best R@1':>11}"
          f"{'max loses':>11}")
    spread_rows = []
    for sg in (1.0, 1.25, 1.5, 2.0):
        rows = []
        for s in range(args.seeds):
            rng = np.random.default_rng(2000 + s)
            r = sweep(rng, (4,), ks, args.n, args.mu, 1, 49, args.pools,
                      sigma_neg=sg)[4]
            rows.append([r[k] for k in ks])
        mean = np.mean(rows, axis=0)
        bi = int(np.argmax(mean))
        spread_rows.append(dict(sigma_neg=sg, max=float(mean[0]),
                                best_k=ks[bi], best=float(mean[bi])))
        print(f"  {sg:>10.2f}{100*mean[0]:>12.1f}{ks[bi]:>9}{100*mean[bi]:>11.1f}"
              f"{100*(mean[bi]-mean[0]):>11.1f}")

    with open(args.out, "w") as f:
        json.dump(dict(config=vars(args), ks=list(ks), by_m=tables,
                       endpoints=endpoint_rows, spread=spread_rows), f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
