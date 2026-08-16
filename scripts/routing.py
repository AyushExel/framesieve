"""Should every query get the same expensive budget?

Everything so far spends a fixed K expensive calls on every query. That is the
same mistake as uniform frame sampling, one level up: it is unbiased, it needs no
extra machinery, and it ignores information that is already sitting there for
free.

MomentSeeker's queries are not one population. Some are captions -- "A man cuts a
piece of wood with a saw, then picks it up and walks away" -- which is exactly
the form a caption-trained encoder was built for. Some are questions -- "Are
there any irregularities in the movements captured by this surveillance
footage?" -- which contain almost no visual content to match. The cheap stage
should be excellent on the first kind and useless on the second, and if so, the
expensive budget is being spent in the wrong places.

Two things have to be true for adaptive routing to be worth anything, and they
are separate claims:

  1. the cheap stage's accuracy really does vary a lot across queries
  2. something FREE predicts which is which -- free meaning computable from the
     score vector the cheap stage already produced, with no extra model call and
     no knowledge of the answer

Candidate free signals, all functions of the ranked chunk scores:

  margin     top1 - top2, normalised by the video's score spread. A clear winner
             means the ranking has an opinion.
  gap5       top1 - top5, same normalisation: is there a peak, or a plateau?
  entropy    entropy of softmax(scores). Low entropy = concentrated belief.
  ntop       how many chunks score within 10% of the top -- a plateau count.

If none of them separates the hits from the misses, adaptive routing has no
signal to route on and the honest answer is to keep the budget flat.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from framesieve.benchmarks.momentseeker import (  # noqa: E402
    chunks_for,
    gt_chunk_mask,
    load_queries,
    recall_at_k,
)
from framesieve.index import FrameIndex  # noqa: E402

TOPK = 4


def chunk_scores(ts: np.ndarray, sims: np.ndarray, ch: np.ndarray,
                 k: int = TOPK) -> np.ndarray:
    ci = np.clip(np.searchsorted(ch[:, 0], ts, "right") - 1, 0, len(ch) - 1)
    order = np.argsort(ci, kind="stable")
    s = sims[order]
    bounds = np.searchsorted(ci[order], np.arange(len(ch) + 1))
    out = np.full(len(ch), -1e9)
    for c in range(len(ch)):
        v = s[bounds[c]:bounds[c + 1]]
        if v.size:
            out[c] = np.sort(v)[-min(k, v.size):].mean()
    return out


def confidence(cs: np.ndarray) -> dict:
    """Free signals: everything here is a function of the score vector alone."""
    v = cs[cs > -1e8]
    if v.size < 3:
        return dict(margin=0.0, gap5=0.0, entropy=0.0, ntop=1.0)
    s = np.sort(v)[::-1]
    sd = v.std() + 1e-9
    p = np.exp((v - v.max()) / (sd + 1e-9))
    p = p / p.sum()
    rng = (s[0] - s[-1]) + 1e-9
    return dict(
        margin=float((s[0] - s[1]) / sd),
        gap5=float((s[0] - s[min(4, len(s) - 1)]) / sd),
        entropy=float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p))),
        ntop=float(np.mean(s >= s[0] - 0.1 * rng)),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="data/ms_raw/t2v.json")
    ap.add_argument("--video-dir", default="data/ms_videos")
    ap.add_argument("--index-dir", default="runs/ms_index")
    ap.add_argument("--encoder", default="siglip2-base-224")
    ap.add_argument("--out", default="runs/routing.json")
    args = ap.parse_args()

    queries = [q for q in load_queries(args.json, args.video_dir) if q.video_path]
    have = {os.path.splitext(f)[0] for f in os.listdir(args.index_dir)
            if f.endswith(".npz")}
    queries = [q for q in queries if q.video_id in have]
    print(f"{len(queries)} queries\n")

    from framesieve.encoders import SiglipEncoder
    enc = SiglipEncoder(args.encoder)
    qemb = np.concatenate(
        [enc.encode_text([q.text for q in queries[i:i + 256]]).cpu().numpy()
         for i in range(0, len(queries), 256)]).astype(np.float32)
    del enc

    rows = []
    cache: dict = {}
    for n, q in enumerate(queries):
        if q.video_id not in cache:
            if len(cache) > 40:
                cache.clear()
            cache[q.video_id] = FrameIndex.from_npz(
                os.path.join(args.index_dir, f"{q.video_id}.npz"))
        idx = cache[q.video_id]
        ch = chunks_for(float(idx.ts[-1]) + 1.0)
        cs = chunk_scores(idx.ts, idx.emb.astype(np.float32) @ qemb[n], ch)
        ranked = np.argsort(-cs)
        is_gt = gt_chunk_mask(ch, q.gt_intervals)
        # rank of the best true chunk: how deep the expensive stage would have to
        # look before the answer is even a candidate
        pos = np.flatnonzero(is_gt[ranked])
        rows.append(dict(task=q.task, meta=q.meta_task,
                         hit1=recall_at_k(ranked, is_gt, 1),
                         hit5=recall_at_k(ranked, is_gt, 5),
                         hit10=recall_at_k(ranked, is_gt, 10),
                         best_rank=int(pos[0]) if pos.size else -1,
                         n_chunks=int(len(ch)),
                         is_question=q.text.strip().endswith("?"),
                         **confidence(cs)))

    # ---- claim 1: does accuracy vary across query kinds? ------------------
    print("cheap stage by query form")
    print(f"  {'form':<14}{'n':>6}{'R@1':>8}{'R@5':>8}{'R@10':>8}")
    for lab, sel in (("caption", lambda r: not r["is_question"]),
                     ("question", lambda r: r["is_question"])):
        g = [r for r in rows if sel(r)]
        print(f"  {lab:<14}{len(g):>6}{100*np.mean([r['hit1'] for r in g]):>8.1f}"
              f"{100*np.mean([r['hit5'] for r in g]):>8.1f}"
              f"{100*np.mean([r['hit10'] for r in g]):>8.1f}")

    print("\ncheap stage by annotated task")
    print(f"  {'task':<24}{'n':>6}{'R@1':>8}{'R@5':>8}{'R@10':>8}")
    by = defaultdict(list)
    for r in rows:
        by[r["task"]].append(r)
    for t in sorted(by, key=lambda t: -np.mean([r["hit1"] for r in by[t]])):
        g = by[t]
        print(f"  {t:<24}{len(g):>6}{100*np.mean([r['hit1'] for r in g]):>8.1f}"
              f"{100*np.mean([r['hit5'] for r in g]):>8.1f}"
              f"{100*np.mean([r['hit10'] for r in g]):>8.1f}")

    # ---- claim 2: does anything free predict a hit? -----------------------
    print("\ndoes a free signal separate the hits from the misses?")
    hit = np.array([r["hit10"] for r in rows], bool)   # can the VLM even see it
    print(f"  {'signal':<12}{'when reachable':>16}{'when not':>12}"
          f"{'AUC':>8}{'verdict':>10}")
    aucs = {}
    for key in ("margin", "gap5", "entropy", "ntop"):
        x = np.array([r[key] for r in rows], float)
        a, b = x[hit], x[~hit]
        # AUC by rank-sum, which needs no binning and no threshold
        rank = np.argsort(np.argsort(x)) + 1
        auc = ((rank[hit].sum() - hit.sum() * (hit.sum() + 1) / 2)
               / (hit.sum() * (~hit).sum())) if hit.any() and (~hit).any() else 0.5
        aucs[key] = float(auc)
        v = "useful" if abs(auc - 0.5) > 0.08 else "no signal"
        print(f"  {key:<12}{a.mean():>16.3f}{b.mean():>12.3f}{auc:>8.3f}{v:>10}")

    best = max(aucs, key=lambda k: abs(aucs[k] - 0.5))
    print(f"\n  strongest: {best} at AUC {aucs[best]:.3f}")
    if abs(aucs[best] - 0.5) <= 0.08:
        print("  VERDICT: nothing free predicts which queries the cheap stage "
              "gets right.\n  Adaptive routing has no signal to route on. "
              "Keep the budget flat.")
    else:
        print("  VERDICT: there is a usable signal -- see the routing simulation "
              "below.")

        # ---- what routing on it would buy, at matched total cost ----------
        # The expensive stage re-ranks the top k_i chunks of query i. Give it the
        # benefit of the doubt -- assume it always picks the true chunk when the
        # true chunk is among the k it was shown -- so R@1 for query i is exactly
        # 1[best_rank_i < k_i], and the whole question becomes how to split a
        # fixed total budget B = sum(k_i) across queries.
        #
        # A flat split gives everyone B/n. A routed split should give more to
        # queries where a few more calls would actually change the answer, and
        # that set is NOT the low-confidence tail: a query whose answer is at
        # rank 400 is unreachable at any sane budget, and one already at rank 0
        # has nothing to gain. The useful queries are in the middle, so the
        # allocation must be allowed to be non-monotone in the signal.
        #
        # The bin -> gain curve is FITTED ON HELD-OUT QUERIES. Fitting it on the
        # same queries it is evaluated on would be reading the answer key.
        br = np.array([r["best_rank"] for r in rows])
        sig = np.array([r[best] for r in rows], float)
        n, KMAX = len(rows), 64
        fold = np.arange(n) % 2

        # The marginal value of query i's k-th expensive call is exactly
        # P(best_rank_i == k-1): the call pays off only if it is the one that
        # first reaches the answer. Routing can only win if that curve has a
        # different SHAPE in different confidence bins -- a bin that is simply
        # better everywhere gives the allocator nothing to trade.
        print("\n  where the k-th call pays off, by " + best + " quartile")
        print(f"  {'quartile':<20}{'P(rank 0)':>11}{'ranks 1-3':>11}"
              f"{'4-9':>8}{'10-31':>8}{'>=32 or unreachable':>22}")
        qe = np.quantile(sig, [0.25, 0.5, 0.75])
        qb = np.digitize(sig, qe)
        buckets = [(0, 1), (1, 4), (4, 10), (10, 32)]
        for b_ in range(4):
            m = qb == b_
            cells = [np.mean((br[m] >= lo) & (br[m] < hi)) for lo, hi in buckets]
            rest = 1.0 - sum(cells)
            # low entropy means a concentrated ranking, so which end is the
            # confident one depends on the sign of the signal's AUC
            conf_hi = aucs[best] > 0.5
            tag = (" most confident" if b_ == (3 if conf_hi else 0) else
                   " least confident" if b_ == (0 if conf_hi else 3) else "")
            print(f"  {'Q'+str(b_+1)+tag:<20}" +
                  "".join(f"{100*c:>11.1f}" for c in cells[:2]) +
                  "".join(f"{100*c:>8.1f}" for c in cells[2:]) +
                  f"{100*rest:>22.1f}")

        def hits_for(alloc):
            return float(np.mean((br >= 0) & (br < alloc)))

        print("\n  fixed total budget, split flat vs routed on " + best)
        print("  (oracle re-ranker: a query is answered iff its true chunk is "
              "among the k it is shown)")
        print(f"  {'calls/query':>13}{'flat R@1':>11}{'routed R@1':>13}{'gain':>8}"
              f"{'spent':>9}")
        table = []
        for mean_k in (2, 4, 8, 16):
            B = mean_k * n
            alloc = np.zeros(n, dtype=int)
            for f in (0, 1):
                tr, te = fold != f, fold == f
                edges = np.quantile(sig[tr], np.linspace(0, 1, 11)[1:-1])
                bt, be = np.digitize(sig[tr], edges), np.digitize(sig[te], edges)
                curve = np.zeros((10, KMAX + 1))
                for bb in range(10):
                    m = bt == bb
                    if m.any():
                        curve[bb] = [((br[tr][m] >= 0) & (br[tr][m] < k)).mean()
                                     for k in range(KMAX + 1)]
                # every query in a bin has the same predicted curve, so the
                # allocation is a choice of k per bin. Greedy on raw marginal
                # gains stalls on the flat stretches these curves have, so take
                # the concave (upper) envelope first -- that makes greedy exact.
                env = np.empty_like(curve)
                for bb in range(10):
                    y, best_so_far = curve[bb], -np.inf
                    slope = np.full(KMAX + 1, -np.inf)
                    # upper concave envelope by scanning slopes from k=0
                    hull = [0]
                    for k in range(1, KMAX + 1):
                        while len(hull) >= 2 and (
                            (y[k] - y[hull[-2]]) / (k - hull[-2])
                            >= (y[hull[-1]] - y[hull[-2]]) / (hull[-1] - hull[-2])
                        ):
                            hull.pop()
                        hull.append(k)
                    env[bb] = np.interp(np.arange(KMAX + 1), hull, y[hull])
                    del best_so_far, slope
                idx_te = np.flatnonzero(te)
                a = np.zeros(len(idx_te), dtype=int)
                gain = env[be, 1] - env[be, 0]
                budget = B // 2
                while budget > 0 and np.max(gain) > 0:
                    j = int(np.argmax(gain))
                    a[j] += 1
                    budget -= 1
                    gain[j] = (env[be[j], a[j] + 1] - env[be[j], a[j]]
                               if a[j] < KMAX else -1.0)
                # any budget the allocator declines to spend is handed out flat,
                # so the comparison is at genuinely matched cost
                if budget > 0:
                    a += budget // len(a)
                alloc[idx_te] = a
            flat, routed = hits_for(np.full(n, mean_k)), hits_for(alloc)
            table.append(dict(mean_k=mean_k, flat=100 * flat, routed=100 * routed,
                              spent=int(alloc.sum()), budget=int(B)))
            print(f"  {mean_k:>13}{100*flat:>11.2f}{100*routed:>13.2f}"
                  f"{100*(routed-flat):>+8.2f}{alloc.sum():>9}")
        gains = [t["routed"] - t["flat"] for t in table]
        print("\n  " + ("routing beats a flat split at every budget"
                        if min(gains) > 0.5 else
                        "routing does NOT beat a flat split: confidence predicts "
                        "whether the answer\n  is reachable, but every bin's "
                        "payoff is front-loaded in the same place, so\n  there is "
                        "no trade to make."))

    with open(args.out, "w") as f:
        json.dump(dict(auc=aucs, routing=table if "table" in dir() else None,
                       rows=rows), f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
