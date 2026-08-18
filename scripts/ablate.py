"""Ablations: which component is actually doing the work?

A cascade has several moving parts and it is easy to credit the wrong one. Each
ablation here changes exactly one thing and re-runs the same evaluation against
the same ground truth:

  factor     segments per VLM call for the budget-adaptive variant. At 1 you take
             one frame from every segment, which is uniform sampling in content
             space; at large values the segments stop constraining anything and
             it reverts to plain top-k. The optimum is in between.
  collapse   segment_tau: 0 (off) .. 0.95. Isolates redundancy collapse from
             selection -- with tau=0 every frame is its own segment, so the
             `segment` strategy degenerates to top-k and any gap between them is
             attributable to the collapse alone.
  encoder    which cheap model does the dense pass. Changes retrieval quality
             without touching selection or the VLM.
  gate       the pixel gate: does skipping near-duplicate frames before the
             encoder cost recall?
  depth      cascade depth: retrieval only (no VLM), retrieval -> VLM, and
             retrieval -> cheap VLM -> expensive VLM.

Every run rebuilds only what it must and reuses the same ground truth, so the
numbers are comparable by construction.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from framesieve.encoders import SIGLIP_MODELS, SiglipEncoder  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _indexio import read_index  # noqa: E402
from framesieve.evaluate import (  # noqa: E402
    bootstrap_ci,
    evaluate_selection,
    events_from_scores,
)
from framesieve.indexing import FrameIndex, build_index  # noqa: E402
from framesieve.search import select_candidates  # noqa: E402


def load_gt(path: str):
    z = np.load(path, allow_pickle=True)
    return z["ts"], z["scores"], [str(q) for q in z["queries"]], json.loads(str(z["meta"]))


def eval_index(idx: FrameIndex, qembs: np.ndarray, gt_ts, gt_scores, events_by_q,
               budgets, strategies, seeds, threshold) -> list[dict]:
    rows = []
    for strat in strategies:
        ns = seeds if strat == "uniform" else 1
        for b in budgets:
            if b > len(idx.ts):
                continue
            per_q = []
            for qi in range(len(events_by_q)):
                ev = events_by_q[qi]
                if not ev:
                    continue
                acc = []
                for sd in range(ns):
                    cand = select_candidates(idx, qembs[qi], b, strategy=strat, seed=sd)
                    er, _, _, _ = evaluate_selection(gt_ts, gt_scores[:, qi], cand.ts,
                                                     ev, threshold=threshold)
                    acc.append(er)
                per_q.append(float(np.mean(acc)))
            m, lo, hi = bootstrap_ci(np.array(per_q))
            rows.append(dict(strategy=strat, budget=b, event_recall=m,
                             event_recall_lo=lo, event_recall_hi=hi,
                             per_query_event_recall=per_q))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/glasgow_mallaig.mp4")
    ap.add_argument("--gt", default="runs/groundtruth_glasgow.npz")
    ap.add_argument("--which", nargs="*",
                    default=["collapse", "factor", "encoder", "gate"])
    ap.add_argument("--budgets", type=int, nargs="*", default=[8, 32, 128, 512])
    ap.add_argument("--strategies", nargs="*", default=["uniform", "topk", "segment"])
    ap.add_argument("--taus", type=float, nargs="*", default=[0.0, 0.85, 0.90, 0.95, 0.98])
    ap.add_argument("--encoders", nargs="*",
                    default=["siglip2-base-224", "siglip2-base-384", "siglip2-so400m-384"])
    ap.add_argument("--gates", type=float, nargs="*", default=[0.0, 1.0, 2.0, 4.0])
    ap.add_argument("--factors", type=float, nargs="*",
                    default=[1.0, 2.0, 4.0, 8.0, 16.0, 64.0])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--queries", default="configs/queries_glasgow.json",
                    help="caption forms for the cheap encoder; must match "
                         "eval_recall_curve.py or the numbers are not comparable")
    ap.add_argument("--cache-dir", default="runs/ablate_idx")
    ap.add_argument("--out", default="runs/ablations.json")
    args = ap.parse_args()

    gt_ts, gt_scores, queries, meta = load_gt(args.gt)
    events_by_q = {qi: events_from_scores(gt_ts, gt_scores[:, qi],
                                          threshold=args.threshold, merge_gap_s=3.0)
                   for qi in range(len(queries))}
    n_ev = sum(len(v) for v in events_by_q.values())
    print(f"ground truth: {len(gt_ts):,} frames, {len(queries)} queries, "
          f"{n_ev} events\n")
    os.makedirs(args.cache_dir, exist_ok=True)
    dur = float(gt_ts[-1]) + 1.0
    out: dict = {"config": vars(args), "queries": queries,
                 "n_events": {q: len(events_by_q[i]) for i, q in enumerate(queries)},
                 "ablations": {}}

    def _restrict(idx: FrameIndex) -> FrameIndex:
        """Keep only the frames the ground truth actually covers.

        Without this, a partial ground-truth run silently scores selections made
        outside its range against its boundary frame. evaluate_selection now
        raises rather than snapping, so this is belt and braces."""
        keep = idx.ts <= gt_ts[-1] + 1e-6
        if keep.all():
            return idx
        return FrameIndex(idx.ts[keep], idx.emb[keep], idx.seg_id[keep], idx.stats)

    def get_index(enc_key: str, tau: float, gate: float) -> tuple[FrameIndex, dict]:
        path = os.path.join(args.cache_dir,
                            f"{enc_key}_tau{tau:g}_gate{gate:g}.npz")
        if os.path.exists(path):
            return _restrict(read_index(path)), {"cached": True}
        enc = SiglipEncoder(enc_key)
        t0 = time.perf_counter()
        idx = build_index(args.video, enc, target_fps=1.0, segment_tau=tau,
                          pixel_gate_tau=gate, duration_s=dur, verbose=False)
        idx.save(path)
        idx = _restrict(idx)
        info = {"build_s": time.perf_counter() - t0,
                "n_segments": idx.stats.n_segments,
                "n_encoded": idx.stats.n_encoded,
                "realtime_factor": idx.stats.realtime_factor}
        del enc
        import torch
        torch.cuda.empty_cache()
        return idx, info

    # The cheap encoder is a caption model. Handing it the ground truth's yes/no
    # question form instead of a caption costs an order of magnitude of recall at
    # small budgets, so every ablation must use the same surface form as the main
    # recall curve or none of the numbers are comparable to it.
    retrieval_text = list(queries)
    if os.path.exists(args.queries):
        spec = json.load(open(args.queries))["queries"]
        by_q = {s["question"]: s["caption"] for s in spec}
        n_mapped = sum(1 for q in queries if q in by_q)
        retrieval_text = [by_q.get(q, q) for q in queries]
        print(f"retrieval form: caption ({n_mapped}/{len(queries)} mapped)\n")
    else:
        print(f"retrieval form: question ({args.queries} not found)\n")

    def qemb_for(enc_key: str) -> np.ndarray:
        enc = SiglipEncoder(enc_key)
        e = enc.encode_text(retrieval_text).cpu().numpy().astype(np.float32)
        del enc
        import torch
        torch.cuda.empty_cache()
        return e

    # ---- redundancy collapse -------------------------------------------
    if "collapse" in args.which:
        print("=== ablation: redundancy collapse (segment_tau) ===")
        qe = qemb_for("siglip2-base-224")
        res = []
        for tau in args.taus:
            idx, info = get_index("siglip2-base-224", tau, 0.0)
            rows = eval_index(idx, qe, gt_ts, gt_scores, events_by_q,
                              args.budgets, args.strategies, args.seeds,
                              args.threshold)
            res.append({"segment_tau": tau, "n_segments": idx.stats.n_segments,
                        "collapse_ratio": len(idx.ts) / max(1, idx.stats.n_segments),
                        "index_info": info, "rows": rows})
            seg = [r for r in rows if r["strategy"] == "segment"]
            print(f"  tau={tau:<5} segments={idx.stats.n_segments:>6,} "
                  f"({len(idx.ts)/max(1,idx.stats.n_segments):>6.1f}x collapse)  "
                  + "  ".join(f"K={r['budget']}:{r['event_recall']:.3f}" for r in seg),
                  flush=True)
        out["ablations"]["collapse"] = res

    # ---- encoder --------------------------------------------------------
    if "encoder" in args.which:
        print("\n=== ablation: cheap encoder ===")
        res = []
        for key in args.encoders:
            if key not in SIGLIP_MODELS:
                continue
            idx, info = get_index(key, 0.90, 0.0)
            qe = qemb_for(key)
            rows = eval_index(idx, qe, gt_ts, gt_scores, events_by_q, args.budgets,
                              args.strategies, args.seeds, args.threshold)
            res.append({"encoder": key, "index_info": info, "rows": rows,
                        "n_segments": idx.stats.n_segments})
            seg = [r for r in rows if r["strategy"] == "segment"]
            print(f"  {key:<20} {info.get('realtime_factor', 0):>6.0f}x realtime  "
                  + "  ".join(f"K={r['budget']}:{r['event_recall']:.3f}" for r in seg),
                  flush=True)
        out["ablations"]["encoder"] = res

    # ---- segments per VLM call -----------------------------------------
    if "factor" in args.which:
        print("\n=== ablation: segments per VLM call (segment_adaptive) ===")
        qe = qemb_for("siglip2-base-224")
        idx, _ = get_index("siglip2-base-224", 0.90, 0.0)
        res = []
        for fac in args.factors:
            rows = []
            for b in args.budgets:
                per_q = []
                for qi in range(len(queries)):
                    ev = events_by_q[qi]
                    if not ev:
                        continue
                    cand = select_candidates(idx, qe[qi], b,
                                             strategy="segment_adaptive",
                                             segment_factor=fac)
                    er, _, _, _ = evaluate_selection(gt_ts, gt_scores[:, qi],
                                                     cand.ts, ev,
                                                     threshold=args.threshold)
                    per_q.append(er)
                m, lo, hi = bootstrap_ci(np.array(per_q))
                rows.append(dict(strategy="segment_adaptive", budget=b,
                                 event_recall=m, event_recall_lo=lo,
                                 event_recall_hi=hi,
                                 per_query_event_recall=per_q))
            res.append({"segment_factor": fac, "rows": rows})
            print(f"  factor={fac:<6} "
                  + "  ".join(f"K={r['budget']}:{r['event_recall']:.3f}" for r in rows),
                  flush=True)
        out["ablations"]["factor"] = res

    # ---- pixel gate -----------------------------------------------------
    if "gate" in args.which:
        print("\n=== ablation: pixel gate (skip near-duplicate frames) ===")
        qe = qemb_for("siglip2-base-224")
        res = []
        for gate in args.gates:
            idx, info = get_index("siglip2-base-224", 0.90, gate)
            rows = eval_index(idx, qe, gt_ts, gt_scores, events_by_q, args.budgets,
                              args.strategies, args.seeds, args.threshold)
            skipped = 1 - idx.stats.n_encoded / max(1, idx.stats.n_frames)
            res.append({"pixel_gate_tau": gate, "frac_skipped": skipped,
                        "index_info": info, "rows": rows})
            seg = [r for r in rows if r["strategy"] == "segment"]
            print(f"  gate={gate:<5} skipped={skipped*100:>5.1f}% of encoder calls  "
                  + "  ".join(f"K={r['budget']}:{r['event_recall']:.3f}" for r in seg),
                  flush=True)
        out["ablations"]["gate"] = res

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
