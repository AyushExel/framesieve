"""The headline experiment: what does the cascade miss, as a function of compute?

Against the dense-VLM ground truth, sweep every selection strategy across a range
of VLM budgets and report event recall with error bars.

What counts as a hit: a selected frame that the VLM confirms. Because ground
truth is the same VLM on the same frames at the same settings, confirmation is a
lookup rather than a second opinion -- see the note in evaluate.py and the check
in verify.py.

Variance, honestly:
  - `uniform` has a random phase, so it gets `--seeds` independent draws and we
    report the spread across them. This matters: uniform sampling's luck is the
    whole reason it is a stronger baseline than people expect.
  - the index-based strategies are deterministic given the index. Their spread is
    across *queries*, reported as a bootstrap CI. We do not manufacture seed
    variance for them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _indexio import read_index  # noqa: E402
from framesieve.evaluate import (  # noqa: E402
    bootstrap_ci,
    evaluate_selection,
    events_from_scores,
)
from framesieve.index import FrameIndex  # noqa: E402
from framesieve.search import select_candidates  # noqa: E402

# measured on this machine, bench/vlm_bench.py, batch 16 @ native visual tokens
VLM_S_PER_FRAME = 0.1072
STOCHASTIC = {"uniform"}


def load_gt(path: str):
    z = np.load(path, allow_pickle=True)
    queries = [str(q) for q in z["queries"]]
    meta = json.loads(str(z["meta"]))
    return z["ts"], z["scores"], queries, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="runs/groundtruth_glasgow.npz")
    ap.add_argument("--index", default="runs/index_glasgow_siglip2b224.npz")
    ap.add_argument("--strategies", nargs="*",
                    default=["uniform", "topk", "nms", "segment"])
    ap.add_argument("--budgets", type=int, nargs="*",
                    default=[4, 8, 16, 32, 64, 128, 256, 512, 1024])
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--merge-gap-s", type=float, default=3.0)
    ap.add_argument("--nms-window-s", type=float, default=None,
                    help="fixed suppression window; default adapts to the budget")
    ap.add_argument("--encoder", default="siglip2-base-224")
    ap.add_argument("--queries", default="configs/queries_glasgow.json",
                    help="caption forms for the cheap encoder; the ground truth's "
                         "question forms are used for the VLM")
    ap.add_argument("--retrieval-form", default="caption",
                    choices=["caption", "question"],
                    help="which surface form the cheap encoder sees; 'question' "
                         "exists so the handicap can be measured, not hidden")
    ap.add_argument("--out", default="runs/recall_curve.json")
    args = ap.parse_args()

    gt_ts, gt_scores, queries, meta = load_gt(args.gt)
    index = read_index(args.index)
    print(f"ground truth : {len(gt_ts):,} frames x {len(queries)} queries "
          f"({gt_ts[0]:.0f}-{gt_ts[-1]:.0f} s)")
    print(f"index        : {len(index.ts):,} frames, "
          f"{index.stats.n_segments:,} segments, {index.stats.encoder}")

    # the index may cover more of the video than a partial ground-truth run
    n = min(len(gt_ts), len(index.ts))
    keep = index.ts <= gt_ts[-1] + 1e-6
    idx = FrameIndex(index.ts[keep], index.emb[keep], index.seg_id[keep], index.stats)
    print(f"evaluating on the {keep.sum():,} frames both cover\n")

    # The cheap encoder is a caption model; the ground truth's queries are yes/no
    # questions written for the VLM. Feeding the question form to SigLIP costs
    # retrieval real accuracy, so the two stages get the surface form each was
    # trained for. The mapping is fixed in configs/ and was not revised after
    # seeing any result.
    retrieval_text = list(queries)
    if os.path.exists(args.queries) and args.retrieval_form == "caption":
        spec = json.load(open(args.queries))["queries"]
        by_q = {s["question"]: s["caption"] for s in spec}
        missing = [q for q in queries if q not in by_q]
        if missing:
            print(f"  warning: no caption for {len(missing)} queries, "
                  f"using the question form for those")
        retrieval_text = [by_q.get(q, q) for q in queries]
        print(f"retrieval form: caption ({len(queries)-len(missing)}/{len(queries)} "
              f"mapped)\n")
    else:
        print(f"retrieval form: {args.retrieval_form}\n")

    from framesieve.encoders import SiglipEncoder
    enc = SiglipEncoder(args.encoder)
    qembs = enc.encode_text(retrieval_text).cpu().numpy().astype(np.float32)

    # ---- ground-truth event structure ------------------------------------
    print(f"{'query':<52}{'pos frames':>11}{'events':>8}{'med len':>9}{'prevalence':>12}")
    print("-" * 92)
    events_by_q = {}
    for qi, q in enumerate(queries):
        sc = gt_scores[:, qi]
        ev = events_from_scores(gt_ts, sc, threshold=args.threshold,
                                merge_gap_s=args.merge_gap_s)
        events_by_q[qi] = ev
        npos = int((sc > args.threshold).sum())
        medlen = float(np.median([e.duration_s + 1 for e in ev])) if ev else 0.0
        print(f"{q[:50]:<52}{npos:>11,}{len(ev):>8}{medlen:>9.1f}"
              f"{100*npos/len(gt_ts):>11.2f}%")
    print()

    # ---- sweep -----------------------------------------------------------
    rows = []
    for strat in args.strategies:
        n_seeds = args.seeds if strat in STOCHASTIC else 1
        for budget in args.budgets:
            if budget > len(idx.ts):
                continue
            per_query_er, per_query_fr, per_query_pr = [], [], []
            seed_means = []
            n_sel_seen = []
            for seed in range(n_seeds):
                ers, frs, prs = [], [], []
                for qi in range(len(queries)):
                    ev = events_by_q[qi]
                    if not ev:
                        continue
                    cand = select_candidates(idx, qembs[qi], budget, strategy=strat,
                                             nms_window_s=args.nms_window_s, seed=seed)
                    n_sel_seen.append(len(cand.ts))
                    er, fr, pr, _ = evaluate_selection(
                        gt_ts, gt_scores[:, qi], cand.ts, ev, threshold=args.threshold)
                    ers.append(er); frs.append(fr); prs.append(pr)
                seed_means.append(float(np.mean(ers)) if ers else float("nan"))
                if seed == 0:
                    per_query_er, per_query_fr, per_query_pr = ers, frs, prs
                else:
                    per_query_er = [a + b for a, b in zip(per_query_er, ers)]
                    per_query_fr = [a + b for a, b in zip(per_query_fr, frs)]
                    per_query_pr = [a + b for a, b in zip(per_query_pr, prs)]
            k = max(1, n_seeds)
            per_query_er = [x / k for x in per_query_er]
            per_query_fr = [x / k for x in per_query_fr]
            per_query_pr = [x / k for x in per_query_pr]

            mean_er, lo, hi = bootstrap_ci(np.array(per_query_er))
            rows.append(dict(
                strategy=strat, budget=budget, n_seeds=n_seeds,
                event_recall=mean_er, event_recall_lo=lo, event_recall_hi=hi,
                event_recall_seed_std=float(np.std(seed_means)) if n_seeds > 1 else 0.0,
                frame_recall=float(np.mean(per_query_fr)),
                precision=float(np.mean(per_query_pr)),
                per_query_event_recall=per_query_er,
                # the budget a strategy was *given* is not always the budget it
                # can spend; record what it actually used so a flat curve is
                # never mistaken for a retrieval result
                n_selected_mean=float(np.mean(n_sel_seen)) if n_sel_seen else 0.0,
                saturated=bool(n_sel_seen and np.mean(n_sel_seen) < budget * 0.99),
                vlm_gpu_s=float(np.mean(n_sel_seen)) * VLM_S_PER_FRAME
                if n_sel_seen else 0.0,
                frac_of_video_seen=budget / len(idx.ts)))
            print(f"  {strat:<9} budget {budget:>5}  event recall "
                  f"{mean_er:>6.3f} [{lo:.3f},{hi:.3f}]"
                  + (f" seed sd {rows[-1]['event_recall_seed_std']:.3f}"
                     if n_seeds > 1 else "")
                  + f"   frame recall {rows[-1]['frame_recall']:.3f}"
                  f"   prec {rows[-1]['precision']:.3f}"
                  + ("  [SATURATED at "
                     f"{rows[-1]['n_selected_mean']:.0f}]" if rows[-1]["saturated"] else ""),
                  flush=True)

    payload = dict(config=vars(args), gt_meta=meta, queries=queries,
                   n_events={q: len(events_by_q[i]) for i, q in enumerate(queries)},
                   vlm_s_per_frame=VLM_S_PER_FRAME,
                   index_stats=index.stats.__dict__, rows=rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {args.out}")

    # ---- the comparison that matters -------------------------------------
    print("\nevent recall at matched VLM budget")
    hdr = f"{'budget':>8}" + "".join(f"{s:>18}" for s in args.strategies)
    print(hdr); print("-" * len(hdr))
    by = {(r["strategy"], r["budget"]): r for r in rows}
    for b in args.budgets:
        if not any((s, b) in by for s in args.strategies):
            continue
        line = f"{b:>8}"
        for s in args.strategies:
            r = by.get((s, b))
            line += f"{r['event_recall']:>12.3f}      " if r else f"{'-':>18}"
        print(line)


if __name__ == "__main__":
    main()
