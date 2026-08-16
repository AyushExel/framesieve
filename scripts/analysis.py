"""Theory, and whether the measurements agree with it.

Four analyses that turn the project's numbers into things you can reason with
before you build:

  1. uniform sampling      a closed form for what evenly-spaced sampling finds,
                           checked against 200 seeded runs
  2. cascade speedup       the cost ratio and the filter ratio combine like
                           parallel resistors; amortisation removes one of them
  3. depth to first hit    the rank at which the cheap stage first surfaces a
                           true positive -- the distribution that actually tells
                           you what budget you need
  4. coverage vs relevance whether a question is answered by one moment or by
                           the whole video decides which sampler wins, and the
                           split is predictable from the question text alone
"""

from __future__ import annotations

import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from framesieve.evaluate import events_from_scores  # noqa: E402
from framesieve.index import FrameIndex  # noqa: E402

OUT = {}


# --------------------------------------------------------------------------
# 1. what uniform sampling finds, in closed form
# --------------------------------------------------------------------------
def uniform_theory(gt_path: str, recall_path: str) -> None:
    """Stratified uniform sampling puts one sample in each of K equal bins, at a
    uniform random offset inside it. For an event of length L in a video of N
    frames, the bin width is N/K, so the event is covered by exactly one bin
    whenever L <= N/K, and the sample inside that bin lands on it with
    probability L/(N/K) = KL/N.

        P(find event e)  =  min(1, K * L_e / N)
        E[event recall]  =  (1/E) * sum_e min(1, K * L_e / N)

    No fitting, no free parameters: the event lengths come from the ground truth
    and K is the budget. If the measured curve does not sit on this, something in
    the sampler is wrong.
    """
    z = np.load(gt_path, allow_pickle=True)
    gt_ts, gt_scores = z["ts"], z["scores"]
    queries = [str(q) for q in z["queries"]]
    N = len(gt_ts)

    lens_by_q = {}
    for qi in range(len(queries)):
        ev = events_from_scores(gt_ts, gt_scores[:, qi], threshold=0.0,
                                merge_gap_s=3.0)
        if ev:
            lens_by_q[qi] = np.array([e.duration_s + 1.0 for e in ev])

    rc = json.load(open(recall_path))
    measured = {r["budget"]: r["event_recall"]
                for r in rc["rows"] if r["strategy"] == "uniform"}

    rows = []
    print("uniform sampling: closed form vs 20 seeded runs")
    print(f"  {'budget':>8}{'predicted':>12}{'measured':>11}{'ratio':>9}")
    for K in sorted(measured):
        per_q = [float(np.mean(np.minimum(1.0, K * L / N)))
                 for L in lens_by_q.values()]
        pred = float(np.mean(per_q))
        meas = measured[K]
        rows.append(dict(budget=K, predicted=pred, measured=meas))
        print(f"  {K:>8}{pred:>12.4f}{meas:>11.4f}"
              f"{(meas/pred if pred else float('nan')):>9.2f}")
    OUT["uniform_theory"] = dict(n_frames=int(N), rows=rows)

    # the practical form of the same statement
    K_needed = {}
    for target in (0.5, 0.9):
        for K in range(1, N):
            per_q = [float(np.mean(np.minimum(1.0, K * L / N)))
                     for L in lens_by_q.values()]
            if float(np.mean(per_q)) >= target:
                K_needed[target] = K
                break
    print(f"  to reach 50% event recall, uniform needs {K_needed.get(0.5)} calls; "
          f"90% needs {K_needed.get(0.9)}  (of {N:,} frames)")
    OUT["uniform_theory"]["calls_needed"] = {str(k): v for k, v in K_needed.items()}


# --------------------------------------------------------------------------
# 2. the cascade's speedup, and its two ceilings
# --------------------------------------------------------------------------
def cascade_speedup() -> None:
    """Dense costs N*e. A cascade costs N*c + K*e. So

        S = N*e / (N*c + K*e) = 1 / (c/e + K/N)

    Write R = e/c for the *cost ratio* and F = N/K for the *filter ratio*:

        1/S = 1/R + 1/F        S = R*F / (R + F)

    The two ceilings combine like parallel resistors. Neither one alone sets the
    speedup, and the smaller one dominates without capping it -- which is why
    improving only the filter, on a pipeline whose cheap stage is not cheap
    enough, buys almost nothing.

    Amortisation changes the picture qualitatively. Index once, serve Q queries,
    and the cheap stage's share is divided by Q:

        1/S = 1/(R*Q) + 1/F    ->    S -> F   as Q -> infinity

    With enough queries the cost ratio stops mattering at all and the filter
    ratio is the only ceiling left. That is the argument for indexing.
    """
    R = 107.23 / 0.1325          # measured: VLM ms/frame over SigLIP ms/frame
    N = 16244
    print("\ncascade speedup: S = R*F/(R+F),  R = cost ratio, F = filter ratio")
    print(f"  measured cost ratio R = {R:.0f}")
    print(f"  {'budget K':>9}{'F = N/K':>10}{'S (1 query)':>13}"
          f"{'S (100 queries)':>17}{'ceiling F':>11}")
    rows = []
    for K in (8, 32, 128, 512, 1024):
        F = N / K
        s1 = R * F / (R + F)
        s100 = 1.0 / (1.0 / (R * 100) + 1.0 / F)
        rows.append(dict(budget=K, F=F, S_1=s1, S_100=s100))
        print(f"  {K:>9}{F:>10.0f}{s1:>13.0f}x{s100:>16.0f}x{F:>11.0f}x")
    OUT["cascade_speedup"] = dict(cost_ratio=R, n_frames=N, rows=rows)
    print("  note both columns approach F, not R: past a handful of queries the")
    print("  cheap stage is free and only the filter ratio limits you.")


# --------------------------------------------------------------------------
# 3. how deep the cheap stage makes you go
# --------------------------------------------------------------------------
def depth_to_first_hit(gt_path: str, index_path: str, queries_path: str,
                       encoder: str = "siglip2-base-224") -> None:
    """Aggregate recall hides the thing you actually need: for this query, how
    far down the cheap ranking is the first true positive? That rank *is* the
    budget you must pay. Its distribution across queries is the honest summary of
    a retriever, and it is far more actionable than a mean.
    """
    from framesieve.encoders import SiglipEncoder

    z = np.load(gt_path, allow_pickle=True)
    gt_ts, gt_scores = z["ts"], z["scores"]
    queries = [str(q) for q in z["queries"]]
    idx = FrameIndex.load(index_path)
    keep = idx.ts <= gt_ts[-1] + 1e-6
    idx = FrameIndex(idx.ts[keep], idx.emb[keep], idx.seg_id[keep], idx.stats)

    spec = json.load(open(queries_path))["queries"]
    cap = {s["question"]: s["caption"] for s in spec}
    enc = SiglipEncoder(encoder)
    qe = enc.encode_text([cap.get(q, q) for q in queries]).cpu().numpy().astype(np.float32)

    print("\ndepth to the first true positive in the cheap ranking")
    print(f"  {'query':<46}{'rank':>7}{'of':>8}{'percentile':>12}")
    rows = []
    for qi, q in enumerate(queries):
        pos = gt_scores[:, qi] > 0
        if not pos.any():
            continue
        order = np.argsort(-(idx.emb.astype(np.float32) @ qe[qi]))
        first = int(np.flatnonzero(pos[order])[0]) + 1
        rows.append(dict(query=q, rank_first_hit=first, n_frames=int(len(order)),
                         percentile=100.0 * first / len(order),
                         n_positive=int(pos.sum())))
        print(f"  {q[:44]:<46}{first:>7}{len(order):>8}{100*first/len(order):>11.2f}%")
    r = np.array([x["rank_first_hit"] for x in rows])
    print(f"  median {np.median(r):.0f}, worst {r.max()}  "
          f"-- a budget of {int(np.median(r))} finds something for half the queries")
    OUT["depth_to_first_hit"] = rows


# --------------------------------------------------------------------------
# 4. coverage vs relevance, predicted from the question text
# --------------------------------------------------------------------------
AGGREGATE = re.compile(
    r"\b(how many|how often|count|number of|in what order|the order|sequence|"
    r"first.*then|overall|mainly|main topic|summar|throughout|in total|total "
    r"number|which of the following is not|not mention)\b", re.I)


def coverage_vs_relevance(vmme_path: str, parquet: str, video_dir: str,
                          index_dir: str) -> None:
    """A question answered by *one* moment is a max over frames; a question about
    the whole video is a mean or a count over frames.

    Selecting the top-k most relevant frames is a good estimator of a max and a
    terrible estimator of a mean -- it is maximally biased by construction. So
    relevance-based selection should beat uniform on localised questions and lose
    on aggregate ones. That is a prediction, and the split can be made from the
    question text alone, before looking at any accuracy.
    """
    import pyarrow.parquet as pq

    from framesieve.benchmarks.videomme import load_items

    d = json.load(open(vmme_path))
    rows = d["rows"]
    items = load_items(parquet, video_dir, durations=("long",))
    items = [it for it in items if it.video_path]
    have = {os.path.splitext(f)[0] for f in os.listdir(index_dir) if f.endswith(".npz")}
    items = [it for it in items if it.videoID in have]
    gold = [it.answer for it in items]
    agg = np.array([bool(AGGREGATE.search(it.question)) for it in items])
    print(f"\ncoverage vs relevance on Video-MME long "
          f"({agg.sum()} aggregate / {(~agg).sum()} localised, split from text alone)")

    by = {}
    for r in rows:
        by.setdefault(r["strategy"], []).append(r["preds"])

    def acc(preds, mask):
        v = [1.0 if (p is not None and p == g) else 0.0
             for p, g, m in zip(preds, gold, mask) if m]
        return float(np.mean(v)) if v else float("nan")

    print(f"  {'strategy':<22}{'localised':>12}{'aggregate':>12}{'difference':>13}")
    out = []
    base_loc = float(np.mean([acc(p, ~agg) for p in by["uniform"]]))
    base_agg = float(np.mean([acc(p, agg) for p in by["uniform"]]))
    for s in ("uniform", "topk", "nms", "segment", "segment_adaptive"):
        if s not in by:
            continue
        loc = float(np.mean([acc(p, ~agg) for p in by[s]]))
        ag = float(np.mean([acc(p, agg) for p in by[s]]))
        out.append(dict(strategy=s, localised=loc * 100, aggregate=ag * 100,
                        vs_uniform_localised=(loc - base_loc) * 100,
                        vs_uniform_aggregate=(ag - base_agg) * 100))
        print(f"  {s:<22}{loc*100:>11.2f}%{ag*100:>11.2f}%"
              f"{(loc-ag)*100:>12.2f}")
    print(f"  {'':<22}{'':>12}{'':>12}")
    print(f"  relevance ranking vs uniform: "
          f"{(out[1]['vs_uniform_localised']):+.2f} on localised, "
          f"{(out[1]['vs_uniform_aggregate']):+.2f} on aggregate")
    OUT["coverage_vs_relevance"] = dict(
        n_aggregate=int(agg.sum()), n_localised=int((~agg).sum()), rows=out)





# --------------------------------------------------------------------------
# 5. why an event is missed: not looked at, or not seen?
# --------------------------------------------------------------------------
def why_missed(gt_path: str, index_path: str, queries_path: str,
               encoder: str = "siglip2-base-224") -> None:
    """For every ground-truth event, find the best rank the cheap stage gives to
    any frame inside it. That single number separates the two failure modes:

      rank <= K   the encoder surfaced it; if we still missed it, the budget was
                  spent elsewhere -- an *allocation* failure, fixable by better
                  selection
      rank >> K   the encoder never ranked it near the top at any budget we would
                  plausibly pay -- a *perception* failure, not fixable by
                  selection at all

    The fraction with rank <= K is exactly the ceiling for *global top-k*, which
    is why a diversity-aware selector can and does exceed it: a frame ranked
    5,000th globally can still be the best frame in a highly ranked segment.
    Comparing the two is the cleanest way to see what diversity buys.

    The average is taken per query and then across queries, matching how event
    recall is computed everywhere else. Pooling all events instead would weight
    queries by how many events they happen to have and is not comparable to the
    recall numbers.
    """
    from framesieve.encoders import SiglipEncoder

    z = np.load(gt_path, allow_pickle=True)
    gt_ts, gt_scores = z["ts"], z["scores"]
    queries = [str(q) for q in z["queries"]]
    idx = FrameIndex.load(index_path)
    keep = idx.ts <= gt_ts[-1] + 1e-6
    idx = FrameIndex(idx.ts[keep], idx.emb[keep], idx.seg_id[keep], idx.stats)
    N = len(idx.ts)

    spec = json.load(open(queries_path))["queries"]
    cap = {s["question"]: s["caption"] for s in spec}
    enc = SiglipEncoder(encoder)
    qe = enc.encode_text([cap.get(q, q) for q in queries]).cpu().numpy().astype(np.float32)

    all_best = []
    per_query = []
    for qi, q in enumerate(queries):
        ev = events_from_scores(gt_ts, gt_scores[:, qi], threshold=0.0,
                                merge_gap_s=3.0)
        if not ev:
            continue
        sims = idx.emb.astype(np.float32) @ qe[qi]
        rank = np.empty(N, dtype=np.int64)
        rank[np.argsort(-sims)] = np.arange(1, N + 1)
        best = []
        for e in ev:
            lo = np.searchsorted(idx.ts, e.t_start - 1e-6, "left")
            hi = np.searchsorted(idx.ts, e.t_end + 1e-6, "right")
            if hi > lo:
                best.append(int(rank[lo:hi].min()))
        best = np.array(best)
        all_best.append(best)
        per_query.append(dict(query=q, n_events=len(best),
                              median_best_rank=float(np.median(best)),
                              frac_within_128=float((best <= 128).mean()),
                              frac_within_1024=float((best <= 1024).mean())))
    b = np.concatenate(all_best)
    # per-query then across queries, to match how event recall is computed
    KS = (8, 32, 128, 512, 1024)
    ceiling = {K: float(np.mean([float((x <= K).mean()) for x in all_best]))
               for K in KS}

    rc_path = "runs/recall_curve.json"
    measured = {}
    if os.path.exists(rc_path):
        for r in json.load(open(rc_path))["rows"]:
            measured.setdefault(r["strategy"], {})[r["budget"]] = r["event_recall"]

    print("\nwhy an event is missed: best cheap-stage rank of any frame in it")
    print(f"  {len(b)} events over {N:,} frames, "
          f"{len(all_best)} queries (averaged per query)")
    print(f"    {'budget':>7}{'top-k ceiling':>15}{'top-k actual':>14}"
          f"{'best strategy':>15}{'diversity buys':>16}")
    for K in KS:
        tk = measured.get("topk", {}).get(K, float("nan"))
        best_s = max((measured.get(s, {}).get(K, float("nan"))
                      for s in ("topk", "nms", "segment", "segment_adaptive")),
                     default=float("nan"))
        print(f"    {K:>7}{100*ceiling[K]:>14.1f}%{100*tk:>13.1f}%"
              f"{100*best_s:>14.1f}%{100*(best_s-ceiling[K]):>15.1f}pt")
    print(f"    median best rank {np.median(b):.0f}, "
          f"p90 {np.percentile(b, 90):.0f}, worst {b.max()}")
    frac_deep = float(np.mean([float((x > 1024).mean()) for x in all_best]))
    print(f"    {100*frac_deep:.1f}% of events have no frame in the top 1024 "
          f"({100*1024/N:.1f}% of the video) -- no selector reaches those")
    OUT["why_missed"] = dict(
        n_events=int(len(b)), n_frames=int(N),
        topk_ceiling={str(K): ceiling[K] for K in KS},
        measured=measured,
        frac_events_beyond_1024=frac_deep,
        median_best_rank=float(np.median(b)),
        p90_best_rank=float(np.percentile(b, 90)),
        per_query=per_query)


# --------------------------------------------------------------------------
# 6. recall stratified by how sure the oracle was
# --------------------------------------------------------------------------
def confidence_strata(gt_path: str, index_path: str, queries_path: str,
                      encoder: str = "siglip2-base-224") -> None:
    """Ground truth here is a model, and a model's low-confidence outputs are
    noise. Scoring recall against all of them measures agreement with that noise.

    The ground-truth score is log P(yes) - log P(no), so 0 is a coin flip and 6
    is about 400:1 odds. Sweeping a floor on each event's peak score shows how
    much of the apparent recall ceiling is the retriever failing and how much is
    the oracle being unsure there was anything there.
    """
    from framesieve.encoders import SiglipEncoder
    from framesieve.evaluate import evaluate_selection
    from framesieve.search import select_candidates

    z = np.load(gt_path, allow_pickle=True)
    gt_ts, gt_sc = z["ts"], z["scores"]
    queries = [str(q) for q in z["queries"]]
    idx = FrameIndex.load(index_path)
    spec = json.load(open(queries_path))["queries"]
    cap = {s["question"]: s["caption"] for s in spec}
    enc = SiglipEncoder(encoder)
    qe = enc.encode_text([cap.get(q, q) for q in queries]).cpu().numpy().astype(np.float32)

    KS = (8, 32, 128, 512, 1024)
    STRATA = [(0.0, "any positive"), (2.0, "peak >= 2"), (4.0, "peak >= 4"),
              (6.0, "peak >= 6")]
    out = []
    print("\nevent recall vs how confident the oracle was")
    print(f"  {'stratum':<16}{'events':>8}" + "".join(f"{'K='+str(k):>9}" for k in KS))
    for floor, lab in STRATA:
        ev_by_q = {}
        for qi in range(len(queries)):
            ev = [e for e in events_from_scores(gt_ts, gt_sc[:, qi], threshold=0.0,
                                                merge_gap_s=3.0)
                  if e.peak_score >= floor]
            if ev:
                ev_by_q[qi] = ev
        rec = {}
        for strat in ("uniform", "segment_adaptive"):
            for K in KS:
                per = []
                seeds = range(20) if strat == "uniform" else [0]
                for qi, ev in ev_by_q.items():
                    acc = []
                    for sd in seeds:
                        c = select_candidates(idx, qe[qi], K, strategy=strat, seed=sd)
                        er, _, _, _ = evaluate_selection(gt_ts, gt_sc[:, qi], c.ts, ev)
                        acc.append(er)
                    per.append(float(np.mean(acc)))
                rec[f"{strat}_{K}"] = float(np.mean(per))
        n_ev = sum(len(v) for v in ev_by_q.values())
        out.append(dict(floor=floor, label=lab, n_events=n_ev, **rec))
        print(f"  {lab:<16}{n_ev:>8}" +
              "".join(f"{100*rec[f'segment_adaptive_{k}']:>8.1f}%" for k in KS))
    OUT["confidence_strata"] = out


if __name__ == "__main__":
    uniform_theory("runs/groundtruth_glasgow.npz", "runs/recall_curve.json")
    cascade_speedup()
    depth_to_first_hit("runs/groundtruth_glasgow.npz",
                       "runs/index_glasgow_siglip2b224.npz",
                       "configs/queries_glasgow.json")
    try:
        from huggingface_hub import hf_hub_download
        pqp = hf_hub_download("lmms-lab/Video-MME",
                              "videomme/test-00000-of-00001.parquet",
                              repo_type="dataset")
        coverage_vs_relevance("runs/videomme_long.json", pqp,
                              "data/vmme_long", "runs/vmme_index")
    except Exception as e:  # noqa: BLE001
        print(f"\ncoverage vs relevance skipped: {type(e).__name__}: {e}")

    why_missed("runs/groundtruth_glasgow.npz",
               "runs/index_glasgow_siglip2b224.npz",
               "configs/queries_glasgow.json")

    confidence_strata("runs/groundtruth_glasgow.npz",
                      "runs/index_glasgow_siglip2b224.npz",
                      "configs/queries_glasgow.json")

    with open("runs/analysis.json", "w") as f:
        json.dump(OUT, f, indent=2)
    print("\nwrote runs/analysis.json")
