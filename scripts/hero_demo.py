"""The figure the post is built around.

Find a query where uniform sampling misses the event *entirely* at a realistic
budget and the cascade finds it, then show the evidence: a timeline of where each
strategy chose to look, and the frames it came back with.

The selection is done honestly. Uniform's phase is random, so "uniform misses" is
only a claim if it misses across many seeds -- a single unlucky draw proves
nothing. The script reports the miss *rate* over `--seeds` draws and picks the
example with the strongest, not the prettiest, evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from framesieve.evaluate import evaluate_selection, events_from_scores  # noqa: E402
from framesieve.index import FrameIndex  # noqa: E402
from framesieve.search import select_candidates  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/glasgow_mallaig.mp4")
    ap.add_argument("--gt", default="runs/groundtruth_glasgow.npz")
    ap.add_argument("--index", default="runs/index_glasgow_siglip2b224.npz")
    ap.add_argument("--queries", default="configs/queries_glasgow.json")
    ap.add_argument("--budget", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--encoder", default="siglip2-base-224")
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--min-events", type=int, default=3,
                    help="a query needs at least this many ground-truth events to "
                         "be eligible as the hero; with one or two, 'the cascade "
                         "found it' cannot be distinguished from luck")
    ap.add_argument("--frames-dir", default="figures/hero_frames")
    ap.add_argument("--out", default="runs/hero_demo.json")
    args = ap.parse_args()

    z = np.load(args.gt, allow_pickle=True)
    gt_ts, gt_scores = z["ts"], z["scores"]
    queries = [str(q) for q in z["queries"]]
    idx = FrameIndex.load(args.index)
    keep = idx.ts <= gt_ts[-1] + 1e-6
    idx = FrameIndex(idx.ts[keep], idx.emb[keep], idx.seg_id[keep], idx.stats)

    spec = json.load(open(args.queries))["queries"] if os.path.exists(args.queries) else []
    caption_of = {s["question"]: s["caption"] for s in spec}

    from framesieve.encoders import SiglipEncoder
    enc = SiglipEncoder(args.encoder)
    qembs = enc.encode_text([caption_of.get(q, q) for q in queries]).cpu().numpy()
    qembs = qembs.astype(np.float32)

    duration = float(gt_ts[-1])
    print(f"video {duration/3600:.2f} h, {len(gt_ts):,} indexed frames, "
          f"budget {args.budget} VLM calls "
          f"({100*args.budget/len(gt_ts):.2f}% of frames)\n")

    print(f"{'query':<46}{'events':>7}{'pos':>6}{'uniform miss':>14}"
          f"{'seg-adapt':>10}{'topk':>8}")
    print("-" * 92)
    cands = []
    for qi, q in enumerate(queries):
        ev = events_from_scores(gt_ts, gt_scores[:, qi], threshold=args.threshold,
                                merge_gap_s=3.0)
        if not ev:
            continue
        npos = int((gt_scores[:, qi] > args.threshold).sum())

        # uniform: miss rate across many random phases
        misses = 0
        uni_recalls = []
        for sd in range(args.seeds):
            c = select_candidates(idx, qembs[qi], args.budget, strategy="uniform",
                                  seed=sd)
            er, _, _, _ = evaluate_selection(gt_ts, gt_scores[:, qi], c.ts, ev,
                                             threshold=args.threshold)
            uni_recalls.append(er)
            misses += int(er == 0.0)
        miss_rate = misses / args.seeds

        got = {}
        for strat in ("segment_adaptive", "segment", "topk", "nms"):
            c = select_candidates(idx, qembs[qi], args.budget, strategy=strat)
            er, _, _, _ = evaluate_selection(gt_ts, gt_scores[:, qi], c.ts, ev,
                                             threshold=args.threshold)
            got[strat] = er

        print(f"{q[:44]:<46}{len(ev):>7}{npos:>6}{miss_rate*100:>13.1f}%"
              f"{got['segment_adaptive']:>10.2f}{got['topk']:>8.2f}")
        cands.append(dict(qi=qi, query=q, caption=caption_of.get(q, q),
                          n_events=len(ev), n_positive=npos,
                          uniform_miss_rate=miss_rate,
                          uniform_mean_recall=float(np.mean(uni_recalls)),
                          segment_recall=got["segment_adaptive"],
                          segment_fixed_recall=got["segment"],
                          topk_recall=got["topk"], nms_recall=got["nms"],
                          events=[(e.t_start, e.t_end, e.n_frames) for e in ev]))

    # The strongest example: uniform almost never finds it, the cascade does.
    # Eligibility needs enough events that the result is not a coin flip -- the
    # rule is fixed here rather than chosen after seeing which query looks best.
    eligible = [c for c in cands if c["n_events"] >= args.min_events]
    if not eligible:
        print(f"\n  no query has >= {args.min_events} events; "
              "falling back to all queries")
        eligible = cands
    scored = sorted(eligible,
                    key=lambda c: (c["segment_recall"] - c["uniform_mean_recall"],
                                   c["uniform_miss_rate"]), reverse=True)
    if not scored:
        raise SystemExit("no query has any ground-truth event")
    hero = scored[0]
    print(f"\nhero: {hero['query']!r}")
    print(f"  {hero['n_events']} event(s), {hero['n_positive']} positive frames "
          f"({100*hero['n_positive']/len(gt_ts):.2f}% of the video)")
    print(f"  uniform  misses entirely in {hero['uniform_miss_rate']*100:.1f}% of "
          f"{args.seeds} random phases; mean recall "
          f"{hero['uniform_mean_recall']:.3f}")
    print(f"  segment_adaptive recall {hero['segment_recall']:.3f}")
    print(f"  top-k    recall {hero['topk_recall']:.3f}")

    qi = hero["qi"]
    picks = {}
    for strat in ("uniform", "topk", "nms", "segment_adaptive"):
        c = select_candidates(idx, qembs[qi], args.budget, strategy=strat, seed=0)
        picks[strat] = np.sort(c.ts)

    from framesieve.figures import fig_hero_timeline
    ev_spans = [(e[0], e[1]) for e in hero["events"]]
    med = float(np.median([e[1] - e[0] + 1 for e in hero["events"]]))
    sub = (f"{args.budget} VLM calls over {duration/3600:.1f} h "
           f"({100*args.budget/len(gt_ts):.2f}% of frames)  ·  "
           f"{len(ev_spans)} events, median {med:.0f} s long, "
           f"{100*hero['n_positive']/len(gt_ts):.1f}% of the video  ·  "
           f"grey ticks = events, ★ = a call that landed on one")
    for mode in ("light", "dark"):
        p = fig_hero_timeline(duration, ev_spans, picks, mode=mode,
                              title=f"“{hero['caption']}”",
                              sub=sub, out="figures/hero_timeline.png")
        print(f"  wrote {p}")

    # the frames the cascade actually returned for this event
    os.makedirs(args.frames_dir, exist_ok=True)
    seg_ts = picks["segment_adaptive"]
    hit = np.zeros(len(seg_ts), bool)
    for a, b in ev_spans:
        hit |= (seg_ts >= a - 0.5) & (seg_ts <= b + 0.5)
    if hit.any():
        from PIL import Image

        from framesieve.fetch import FrameFetcher
        _, frames = FrameFetcher(args.video, workers=16).fetch(seg_ts[hit].tolist())
        for t, fr in zip(seg_ts[hit], frames):
            Image.fromarray(fr).save(
                os.path.join(args.frames_dir, f"hit_t{int(t):06d}.jpg"), quality=92)
        print(f"  wrote {int(hit.sum())} hit frames to {args.frames_dir}/")

    with open(args.out, "w") as f:
        json.dump(dict(config=vars(args), candidates=cands, hero=hero,
                       picks={k: v.tolist() for k, v in picks.items()}), f, indent=2)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
