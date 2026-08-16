"""Regenerate every figure in the post from the artifacts in runs/.

One command, so the figures can never drift from the numbers. Each figure is
skipped with a note if its input is missing, rather than failing the whole run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from framesieve import figures as F  # noqa: E402

MODES = ("light", "dark")


def _load(path: str):
    if not os.path.exists(path):
        print(f"  skip: {path} missing")
        return None
    with open(path) as f:
        return json.load(f)


def cost_hierarchy(runs: str) -> None:
    enc = _load(f"{runs}/encode_bench.json")
    vlm = _load(f"{runs}/vlm_bench.json")
    dec = _load(f"{runs}/decode_resolution_sweep.json")
    if not (enc and vlm and dec):
        return
    best = lambda rs: max(rs, key=lambda r: r["frames_per_s"])  # noqa: E731
    pick = lambda rs, k, v: [r for r in rs if r[k] == v]        # noqa: E731
    d1080 = [r for r in dec["results"]
             if "1920x1080" in r["name"] and r["backend"] == "cpu"][0]
    stages = [
        ("decode 1080p (CPU)", 1.0 / d1080["frames_per_s"]),
        ("SigLIP2-base-224", 1.0 / best(pick(enc["results"], "model",
                                             "siglip2-base-224"))["frames_per_s"]),
        ("SigLIP2-so400m-384", 1.0 / best(pick(enc["results"], "model",
                                               "siglip2-so400m-384"))["frames_per_s"]),
        ("Qwen2.5-VL-7B @64 tok", 1.0 / best(pick(vlm["results"], "max_visual_tokens",
                                                  64))["frames_per_s"]),
        ("Qwen2.5-VL-7B @native", 1.0 / best(pick(vlm["results"], "max_visual_tokens",
                                                  256))["frames_per_s"]),
    ]
    for m in MODES:
        print("  " + F.fig_cost_hierarchy(
            stages, mode=m, sub="measured on one GH200; lower is cheaper"))


def decode_scaling(runs: str) -> None:
    dec = _load(f"{runs}/decode_resolution_sweep.json")
    if not dec:
        return
    rows = []
    for r in dec["results"]:
        if not r["ok"]:
            continue
        w, h = r["name"].split()[0].split("x")
        rows.append(dict(backend=r["backend"], pixels=int(w) * int(h),
                         realtime_factor=r["realtime_factor"]))
    for m in MODES:
        print("  " + F.fig_decode_scaling(
            rows, mode=m,
            sub="same 300 s of content re-encoded at five resolutions"))


def redundancy(runs: str) -> None:
    st = _load(f"{runs}/vmme_index_stats.json")
    if not st:
        return
    for m in MODES:
        print("  " + F.fig_redundancy(
            [x["ratio"] for x in st], mode=m,
            sub="300 Video-MME long videos, 205.5 h, segment_tau=0.90"))


def recall_curve(runs: str) -> None:
    rc = _load(f"{runs}/recall_curve.json")
    if not rc:
        return
    # describe the ground truth's actual coverage, not the index's -- against a
    # partial ground-truth run those differ, and quoting the index would overstate
    # what the numbers are computed over
    gt = rc.get("gt_meta", {})
    n = gt.get("n_frames_done") or rc.get("index_stats", {}).get("n_frames", 0)
    hrs = n / 3600.0
    n_ev = sum(rc.get("n_events", {}).values())
    sub = (f"{len(rc['queries'])} queries, {n_ev} events over {hrs:.1f} h "
           f"({n:,} frames at 1 fps); band = 95% bootstrap CI over queries")
    for m in MODES:
        print("  " + F.fig_recall_curve(rc["rows"], mode=m, sub=sub))


def tau_sweep(runs: str) -> None:
    ab = _load(f"{runs}/ablations.json")
    if not ab or "collapse" not in ab.get("ablations", {}):
        print("  skip: no collapse ablation")
        return
    ent = ab["ablations"]["collapse"]
    n_ev = sum(ab.get("n_events", {}).values())
    sub = (f"{len(ab['queries'])} queries, {n_ev} ground-truth events; "
           "the segment strategy only, so the tau is the only thing changing")
    for m in MODES:
        print("  " + F.fig_tau_sweep(ent, mode=m, sub=sub))


def ceiling(runs: str) -> None:
    a = _load(f"{runs}/analysis.json")
    if not a or "why_missed" not in a:
        print("  skip: no ceiling analysis"); return
    w = a["why_missed"]
    sub = (f"{w['n_events']} ground-truth events. The dashed line is the limit the "
           f"cheap ranking imposes on top-k; the band is what diversity adds.")
    for m in MODES:
        print("  " + F.fig_ceiling(w["topk_ceiling"], w["measured"], mode=m, sub=sub))


def confidence(runs: str) -> None:
    a = _load(f"{runs}/analysis.json")
    if not a or "confidence_strata" not in a:
        print("  skip: no confidence analysis"); return
    sub = ("ground truth is a VLM; its score is log P(yes) - log P(no), so 0 is a "
           "coin flip and 6 is about 400:1. Budget-adaptive segment selection.")
    for m in MODES:
        print("  " + F.fig_confidence(a["confidence_strata"], mode=m, sub=sub))


def pareto(runs: str) -> None:
    """Accuracy against compute on Video-MME long, across the budget sweep.

    x is VLM GPU-seconds per question, which is what the budget actually buys.
    Both files are merged so the K=8 point (run with three seeds) sits on the
    same curve as the rest of the sweep.
    """
    vm = _load(f"{runs}/videomme_sweep.json")
    k8 = _load(f"{runs}/videomme_long.json")
    if not vm:
        return
    rows = list(vm["rows"]) + [r for r in (k8 or {}).get("rows", [])
                               if r["strategy"] in ("uniform", "segment_adaptive")]
    agg: dict = {}
    for r in rows:
        key = (r["strategy"], r["budget"])
        agg.setdefault(key, []).append(r)
    pts = []
    for (strat, budget), rs in sorted(agg.items(), key=lambda kv: kv[0][1]):
        pts.append(dict(strategy=strat, budget=budget,
                        vlm_gpu_s=float(np.mean([r["vlm_s"] for r in rs]))
                        / max(1, rs[0]["n_questions"]),
                        accuracy=float(np.mean([r["accuracy"] for r in rs]))))

    for m in MODES:
        print("  " + F.fig_pareto(
            [p for p in pts if p.get("kind") == "reference"
             or np.isfinite(p["vlm_gpu_s"])],
            mode=m, title="Accuracy against compute, Video-MME long",
            sub=("900 long-split questions; every point is a frame budget K, "
                 "from 2 to 32. The two curves lie on top of each other."),
            xlabel="VLM GPU-seconds per question  (log)",
            ylabel="accuracy"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    jobs = {"cost": cost_hierarchy, "decode": decode_scaling,
            "redundancy": redundancy, "recall": recall_curve,
            "tau": tau_sweep, "ceiling": ceiling, "confidence": confidence,
            "pareto": pareto}
    for name, fn in jobs.items():
        if args.only and name not in args.only:
            continue
        print(f"{name}:")
        try:
            fn(args.runs)
        except Exception as e:  # noqa: BLE001
            print(f"  failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
