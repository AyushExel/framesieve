"""Video-MME (long split) under the standard protocol, for every selection strategy.

Protocol, matched to the frame-selection literature:
  - K frames selected per question, passed to the VLM as a frame sequence
  - four-way multiple choice, accuracy, no subtitles
  - the answer is read from the first-token logits over {A,B,C,D}, so a model
    that rambles is not scored as wrong for a parsing reason

The only thing that varies between conditions is *which* K frames get chosen.
Everything downstream -- model, revision, resolution, prompt, decoding -- is
identical, so any accuracy difference is attributable to selection.

No hyperparameter here was chosen by looking at Video-MME. segment_tau comes from
the held-out cab-ride video. This script measures; it does not tune.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from framesieve.benchmarks.videomme import (  # noqa: E402
    accuracy,
    accuracy_by,
    bootstrap_accuracy_ci,
    load_items,
)
from framesieve.encoders import SiglipEncoder  # noqa: E402
from framesieve.fetch import FrameFetcher  # noqa: E402
from framesieve.indexing import FrameIndex  # noqa: E402
from framesieve.search import select_candidates  # noqa: E402
from framesieve.vlm import QwenMultiFrameQA, QwenYesNoScorer  # noqa: E402

STOCHASTIC = {"uniform"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=None)
    ap.add_argument("--video-dir", default="data/vmme_long")
    ap.add_argument("--index-dir", default="runs/vmme_index")
    ap.add_argument("--strategies", nargs="*",
                    default=["uniform", "topk", "nms", "segment"])
    ap.add_argument("--budgets", type=int, nargs="*", default=[8])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--encoder", default="siglip2-base-224")
    ap.add_argument("--vlm", default="qwen2.5-vl-7b")
    ap.add_argument("--tokens-per-frame", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="runs/videomme_long.json")
    args = ap.parse_args()

    if args.parquet is None:
        from huggingface_hub import hf_hub_download
        args.parquet = hf_hub_download("lmms-lab/Video-MME",
                                       "videomme/test-00000-of-00001.parquet",
                                       repo_type="dataset")

    items = load_items(args.parquet, args.video_dir, durations=("long",))
    items = [it for it in items if it.video_path]
    have_idx = {os.path.splitext(f)[0] for f in os.listdir(args.index_dir)
                if f.endswith(".npz")}
    items = [it for it in items if it.videoID in have_idx]
    if args.limit:
        items = items[: args.limit]
    print(f"{len(items)} questions over {len({i.videoID for i in items})} videos "
          f"with indexes available")

    enc = SiglipEncoder(args.encoder)
    px = args.tokens_per_frame * 28 * 28 * 4
    scorer = QwenYesNoScorer(args.vlm, max_pixels=px, min_pixels=min(px, 64 * 28 * 28))
    qa = QwenMultiFrameQA(scorer)
    print(json.dumps(scorer.describe(), indent=2))

    # cache indexes and text embeddings; both are reused across conditions
    index_cache: dict[str, FrameIndex] = {}
    fetch_cache: dict[str, FrameFetcher] = {}

    def get_index(vid: str) -> FrameIndex:
        if vid not in index_cache:
            if len(index_cache) > 8:
                index_cache.clear()
            index_cache[vid] = FrameIndex.from_npz(os.path.join(args.index_dir, f"{vid}.npz"))
        return index_cache[vid]

    def get_fetcher(path: str) -> FrameFetcher:
        if path not in fetch_cache:
            if len(fetch_cache) > 8:
                fetch_cache.clear()
            fetch_cache[path] = FrameFetcher(path, workers=16)
        return fetch_cache[path]

    print("\nencoding question text...", flush=True)
    qtexts = [it.retrieval_query() for it in items]
    qemb = np.concatenate([enc.encode_text(qtexts[i:i + 256]).cpu().numpy()
                           for i in range(0, len(qtexts), 256)]).astype(np.float32)

    results = []
    retrieval_z: dict[int, float] = {}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    for strat in args.strategies:
        n_seeds = args.seeds if strat in STOCHASTIC else 1
        for budget in args.budgets:
            for seed in range(n_seeds):
                preds, golds = [], []
                t0 = time.perf_counter()
                sel_s = fetch_s = vlm_s = 0.0
                for n, it in enumerate(items):
                    idx = get_index(it.videoID)
                    t = time.perf_counter()
                    cand = select_candidates(idx, qemb[n], budget, strategy=strat,
                                             seed=seed)
                    sel_s += time.perf_counter() - t
                    if n not in retrieval_z:
                        # how peaked is the retrieval signal for this question?
                        # Many Video-MME questions ("how many...", "in what
                        # order...", "which is NOT...") are not about a
                        # localisable moment, so no selector can help on them.
                        # Recording this lets the result be reported stratified
                        # rather than averaged into meaninglessness.
                        s = idx.emb.astype(np.float32) @ qemb[n]
                        retrieval_z[n] = float((s.max() - s.mean()) / (s.std() + 1e-9))
                    order = np.argsort(cand.ts)      # VLM wants them in time order
                    ts_sorted = cand.ts[order]

                    t = time.perf_counter()
                    got_ts, frames = get_fetcher(it.video_path).fetch(ts_sorted.tolist())
                    fetch_s += time.perf_counter() - t
                    if len(frames) == 0:
                        preds.append(None); golds.append(it.answer); continue

                    t = time.perf_counter()
                    try:
                        pred = qa.answer_letter_logits(list(frames), it.prompt())
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        pred = None
                    vlm_s += time.perf_counter() - t
                    preds.append(pred); golds.append(it.answer)

                    if (n + 1) % 100 == 0:
                        el = time.perf_counter() - t0
                        print(f"  {strat} K={budget} seed={seed}: {n+1}/{len(items)} "
                              f"acc={accuracy(preds, golds):.3f} "
                              f"({el/(n+1)*1000:.0f} ms/q, eta "
                              f"{(len(items)-n-1)*el/(n+1)/60:.1f} min)", flush=True)

                acc, lo, hi = bootstrap_accuracy_ci(preds, golds)
                row = dict(strategy=strat, budget=budget, seed=seed,
                           accuracy=acc, accuracy_lo=lo, accuracy_hi=hi,
                           n_questions=len(golds),
                           n_unparsed=sum(1 for p in preds if p is None),
                           by_domain={k: v for k, v in
                                      accuracy_by(items, preds, "domain").items()},
                           by_task={k: v for k, v in
                                    accuracy_by(items, preds, "task_type").items()},
                           select_s=sel_s, fetch_s=fetch_s, vlm_s=vlm_s,
                           wall_s=time.perf_counter() - t0,
                           preds=preds,
                           retrieval_z=[retrieval_z.get(i) for i in range(len(items))])
                results.append(row)
                print(f"[{strat} K={budget} seed={seed}] accuracy "
                      f"{acc*100:.2f}% [{lo*100:.2f},{hi*100:.2f}]  "
                      f"select {sel_s:.1f}s fetch {fetch_s:.1f}s vlm {vlm_s:.1f}s",
                      flush=True)
                with open(args.out, "w") as f:
                    json.dump(dict(config=vars(args), vlm=scorer.describe(),
                                   encoder=enc.describe(), rows=results), f, indent=2)

    # ---- summary ---------------------------------------------------------
    print(f"\n{'strategy':<12}{'K':>4}{'accuracy %':>14}{'95% CI':>18}{'ms/question':>14}")
    print("-" * 62)
    agg = defaultdict(list)
    for r in results:
        agg[(r["strategy"], r["budget"])].append(r)
    for (s, b), rs in sorted(agg.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        a = float(np.mean([r["accuracy"] for r in rs]))
        lo = float(np.mean([r["accuracy_lo"] for r in rs]))
        hi = float(np.mean([r["accuracy_hi"] for r in rs]))
        ms = float(np.mean([r["wall_s"] for r in rs])) / max(1, rs[0]["n_questions"]) * 1000
        sd = float(np.std([r["accuracy"] for r in rs])) if len(rs) > 1 else 0.0
        print(f"{s:<12}{b:>4}{a*100:>13.2f}%{f'[{lo*100:.2f},{hi*100:.2f}]':>18}"
              f"{ms:>14.0f}" + (f"   seed sd {sd*100:.2f}" if len(rs) > 1 else ""))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
