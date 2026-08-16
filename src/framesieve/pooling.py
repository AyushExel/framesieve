"""Pooling many fine-grained scores into one, with the depth as an explicit knob.

Standalone: numpy only, no imports from the rest of this package. Copy the file
if that is easier than depending on it.

The operation this covers is everywhere in retrieval. You have n scores for the
sub-units of a candidate -- sentences in a chunk, chunks in a document, token
similarities in a late-interaction score, frames in a video segment -- and you
need one score for the candidate. Almost every system writes `max` or `mean`.

Those are the same function:

    max(x)  == topk_mean(x, k=1)
    mean(x) == topk_mean(x, k=n)

so the real parameter is k, and the measured result behind this module is that
the best k tracks m, the number of sub-units that genuinely match. Correlation of
log k* with log m is +0.996 on a synthetic where m is known by construction and
+0.941 on real video where m is measured by a dense per-frame oracle.

Two things about measuring it, both learned the hard way:

  - use a HEAD metric (R@1, precision@1, success@1). Under AUC the effect
    disappears entirely and the correlation with m falls to +0.37 on the same
    data, because max's failure mode is a single inflated negative taking the top
    slot, which an average over the ranking barely registers.
  - expect a broad optimum. Four unrelated families of statistic all peaked at
    the same value here, and the interior was flat in every one. The gain is in
    leaving the endpoint, not in finding the exact middle. A sharp optimum is
    almost certainly noise.

Typical use:

    from framesieve.pooling import estimate_m, sweep_k, topk_mean

    k = estimate_m(sub_unit_is_relevant)          # from a few labelled examples
    scores = topk_mean(per_subunit_scores, k)     # then just use it

    sweep_k(candidate_scores, is_positive, group) # or check the whole curve
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["topk_mean", "estimate_m", "sweep_k", "recommend_k", "k_range",
           "COUNTED_M_TO_K"]


def topk_mean(x, k: int, axis: int = -1):
    """Mean of the k largest values along `axis`. k=1 is max, k>=n is mean.

    Accepts ragged input as a list of 1-D sequences, since candidates rarely all
    have the same number of sub-units in practice.
    """
    if isinstance(x, (list, tuple)) and len(x) and np.ndim(x[0]) >= 1 \
            and not isinstance(x, np.ndarray):
        try:
            arr = np.asarray(x, dtype=float)
        except (ValueError, TypeError):
            arr = None
        if arr is None or arr.dtype == object:
            return np.array([topk_mean(np.asarray(row, dtype=float), k, -1)
                             for row in x])
        x = arr
    a = np.asarray(x, dtype=float)
    n = a.shape[axis]
    k = int(max(1, min(k, n)))
    if k == 1:
        return a.max(axis=axis)
    if k >= n:
        return a.mean(axis=axis)
    # partition is O(n) where a full sort is O(n log n); for the k largest we
    # only need the split point, not the order within either side
    part = np.partition(a, n - k, axis=axis)
    sl = [slice(None)] * a.ndim
    sl[axis] = slice(n - k, None)
    return part[tuple(sl)].mean(axis=axis)


def estimate_m(relevant, min_examples: int = 10) -> float:
    """Median number of genuinely-matching sub-units in a positive candidate.

    `relevant` is one boolean sequence per POSITIVE candidate, marking which of
    its sub-units actually answer the query. Twenty hand-labelled examples is
    enough; this is a property of your data, not of your model, so it does not
    need a held-out split.

    Returns the median rather than the mean because the distribution is heavily
    right-skewed -- a few candidates match almost everywhere and would drag a
    mean well past where the optimum sits.
    """
    counts = [int(np.sum(np.asarray(r, dtype=bool))) for r in relevant]
    counts = [c for c in counts if c > 0]
    if len(counts) < min_examples:
        raise ValueError(
            f"only {len(counts)} positive examples with a labelled sub-unit; "
            f"estimate_m needs at least {min_examples} to mean anything")
    return float(np.median(counts))


# Measured ratio of the best k to the COUNTED m, across every real dataset in
# this project. The counted number is consistently about twice the useful one,
# because a labelled span is wider than the evidence inside it: an interval
# marked relevant contains frames, sentences or tokens a reader would call part
# of the answer but that carry no signal an encoder can pick up.
#
#   MomentSeeker, 837 positive chunks   counted m 8.0   best k 4    0.50
#   dense-oracle video, m = 2           counted m 2     best k 1    0.50
#   dense-oracle video, m = 3-4         counted m 3.5   best k 2    0.57
#   dense-oracle video, m = 5-7         counted m 6     best k 3    0.50
#   dense-oracle video, m = 8+          counted m 9.5   best k 3    0.32
#
# On the synthetic, where m is exact by construction and the noise is Gaussian,
# the ratio is 1.0-1.5 instead. That difference IS the label-width effect, and
# it is why this returns m/2 rather than m.
COUNTED_M_TO_K = 0.5


def recommend_k(relevant, min_examples: int = 10) -> int:
    """A starting k from labelled examples. Start here, then sweep k_range().

    Returns roughly half the counted m -- see COUNTED_M_TO_K for why, and for the
    measurements behind it. This is calibrated on video retrieval; if your labels
    are tighter than a time interval, the ratio will be closer to 1.
    """
    return max(1, int(round(COUNTED_M_TO_K * estimate_m(relevant, min_examples))))


def k_range(relevant, min_examples: int = 10) -> list:
    """The k values worth sweeping: 1 up to the counted m.

    The optimum has never sat above the counted m in any measurement here, and
    the curve is broad, so this is a small and sufficient search. Returned as a
    list so it can go straight into sweep_k(ks=...).
    """
    m = int(np.ceil(estimate_m(relevant, min_examples)))
    ks = sorted({1, 2, 3, 4, 6, 8, 12, 16, 24, 32} | {m, max(1, m // 2)})
    return [k for k in ks if 1 <= k <= max(2, m)]


def sweep_k(scores: Sequence, is_positive, group=None,
            ks: Sequence[int] = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32),
            metric: str = "head", seed: int = 0,
            n_boot: int = 1000) -> dict:
    """Score every k under a head metric, with a bootstrap interval.

    scores       one sequence of sub-unit scores per candidate
    is_positive  bool per candidate
    group        query id per candidate. Ranking happens WITHIN a group, which is
                 what makes this a retrieval measurement rather than a
                 classification one. Defaults to a single group.
    metric       "head" -> share of groups whose top-ranked candidate is positive
                 "auc"  -> included so you can see it hide the effect, not
                           because you should decide on it

    Returns {k: {"score", "lo", "hi"}} plus "best_k" and "flat" -- flat is True
    when every k inside the interval of the best one, meaning the curve is broad
    and you should not read much into the exact argmax.
    """
    pos = np.asarray(is_positive, dtype=bool)
    grp = np.zeros(len(pos), dtype=np.int64) if group is None \
        else np.asarray(group)
    uniq, ginv = np.unique(grp, return_inverse=True)
    rng = np.random.default_rng(seed)

    out: dict = {}
    for k in ks:
        s = np.asarray(topk_mean(scores, k), dtype=float)
        vals = np.full(len(uniq), np.nan)
        for gi in range(len(uniq)):
            m = ginv == gi
            if not m.any() or not pos[m].any() or pos[m].all():
                continue
            if metric == "head":
                vals[gi] = float(pos[m][int(np.argmax(s[m]))])
            elif metric == "auc":
                x = s[m]
                r = np.argsort(np.argsort(x)) + 1
                a, b = int(pos[m].sum()), int((~pos[m]).sum())
                vals[gi] = (r[pos[m]].sum() - a * (a + 1) / 2) / (a * b)
            else:
                raise ValueError(f"unknown metric {metric!r}")
        v = vals[~np.isnan(vals)]
        if v.size == 0:
            raise ValueError("no group has both a positive and a negative "
                             "candidate, so there is nothing to rank")
        boot = np.array([rng.choice(v, size=v.size, replace=True).mean()
                         for _ in range(n_boot)])
        out[k] = dict(score=float(v.mean()),
                      lo=float(np.percentile(boot, 2.5)),
                      hi=float(np.percentile(boot, 97.5)))

    best_k = max(out, key=lambda k: out[k]["score"])
    lo = out[best_k]["lo"]
    flat = all(out[k]["score"] >= lo for k in ks)
    return dict(by_k=out, best_k=best_k, flat=bool(flat), n_groups=int(v.size),
                metric=metric,
                note=("the curve is flat -- every k is inside the best one's "
                      "interval, so pick anything off the endpoints"
                      if flat else
                      f"k={best_k} is outside the interval of at least one other"))
