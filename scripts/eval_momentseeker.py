"""MomentSeeker (t2v split) under the benchmark's own protocol.

This is the external check that actually matches what framesieve does: text query
in, time interval out, on videos averaging over 500 s.

Two conditions are measured on the same candidate chunks, so the difference is
attributable to the cascade and nothing else:

  retrieval-only   rank all 10 s chunks by the cheap encoder. This is the
                   comparison against the paper's retrieval baselines.
  cascade          take the cheap encoder's top-N chunks, re-rank them with the
                   VLM, return the top 5. Costs N VLM calls per query.

The point of the second is not to win on accuracy at any cost -- it is to show
what a fixed number of expensive-model calls buys on a standard benchmark, which
is the same accuracy-vs-compute question the rest of the project asks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from framesieve.benchmarks.momentseeker import (  # noqa: E402
    CHUNK_S,
    IOU_THRESHOLD,
    PAPER_BASELINES,
    chunks_for,
    gt_chunk_mask,
    load_queries,
    map_at_5,
    map_at_5_matched,
    recall_at_k,
    score_chunks,
)
from framesieve.encoders import CLIP_MODELS, ClipEncoder, SiglipEncoder  # noqa: E402
from framesieve.index import FrameIndex, build_index  # noqa: E402


def bootstrap(vals: np.ndarray, n_boot: int = 2000, seed: int = 0):
    v = np.asarray(vals, float)
    if not len(v):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    b = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(1)
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="data/ms_raw/t2v.json")
    ap.add_argument("--video-dir", default="data/ms_videos")
    ap.add_argument("--index-dir", default="runs/ms_index")
    ap.add_argument("--encoder", default="siglip2-base-224")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--agg", default="topk", choices=["max", "mean", "topk"])
    ap.add_argument("--topk", type=int, default=4,
                    help="frames averaged per chunk when --agg topk")
    ap.add_argument("--vlm-budgets", type=int, nargs="*", default=[0],
                    help="0 = retrieval only; N = re-rank the top-N chunks with the VLM")
    ap.add_argument("--vlm", default="qwen2.5-vl-7b")
    ap.add_argument("--tokens-per-frame", type=int, default=128)
    ap.add_argument("--rerank-frames", type=int, default=1,
                    help="frames per chunk shown to the VLM when re-ranking. 1 "
                         "scores a still; >1 shows the chunk as a short clip, "
                         "which is the only way a query about something that "
                         "happens over time can be verified")
    ap.add_argument("--index-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="runs/momentseeker.json")
    args = ap.parse_args()

    queries = load_queries(args.json, args.video_dir)
    queries = [q for q in queries if q.video_path]
    if args.limit:
        queries = queries[: args.limit]
    vids = sorted({q.video_id for q in queries})
    print(f"{len(queries)} queries over {len(vids)} videos")

    os.makedirs(args.index_dir, exist_ok=True)
    enc = (ClipEncoder if args.encoder in CLIP_MODELS
           else SiglipEncoder)(args.encoder)

    # ---- index every video once ------------------------------------------
    todo = [v for v in vids if not os.path.exists(os.path.join(args.index_dir, f"{v}.npz"))]
    if todo:
        print(f"indexing {len(todo)} videos at {args.fps} fps ...")
        t0 = time.perf_counter()
        vid_s = 0.0
        for i, v in enumerate(todo):
            p = next(q.video_path for q in queries if q.video_id == v)
            try:
                idx = build_index(p, enc, target_fps=args.fps, segment_tau=0.90,
                                  verbose=False)
                idx.save(os.path.join(args.index_dir, f"{v}.npz"))
                vid_s += idx.stats.duration_s
            except Exception as e:  # noqa: BLE001
                print(f"  [FAIL] {v}: {type(e).__name__}: {str(e)[:90]}", flush=True)
            if (i + 1) % 25 == 0:
                el = time.perf_counter() - t0
                print(f"  {i+1}/{len(todo)}  {vid_s/3600:.1f} h of video  "
                      f"{el/60:.1f} min  {vid_s/max(el,1e-9):.0f}x realtime", flush=True)
        el = time.perf_counter() - t0
        print(f"indexed {vid_s/3600:.1f} h in {el/60:.1f} min "
              f"= {vid_s/max(el,1e-9):.0f}x realtime\n")
    if args.index_only:
        return

    # ---- query embeddings -------------------------------------------------
    qemb = np.concatenate([enc.encode_text([q.text for q in queries[i:i + 256]]).cpu().numpy()
                           for i in range(0, len(queries), 256)]).astype(np.float32)

    scorer = qa = None
    if any(b > 0 for b in args.vlm_budgets):
        from framesieve.vlm import QwenYesNoScorer
        px = args.tokens_per_frame * 28 * 28 * 4
        scorer = QwenYesNoScorer(args.vlm, max_pixels=px,
                                 min_pixels=min(px, 64 * 28 * 28))

    cache: dict[str, FrameIndex] = {}

    def get_index(v: str) -> FrameIndex:
        if v not in cache:
            if len(cache) > 12:
                cache.clear()
            cache[v] = FrameIndex.load(os.path.join(args.index_dir, f"{v}.npz"))
        return cache[v]

    all_rows = []
    for budget in args.vlm_budgets:
        r1, m5, m5m, r5 = [], [], [], []
        by_task: dict[str, list] = {}
        t0 = time.perf_counter()
        vlm_calls = 0
        t_select = t_fetch = t_vlm = 0.0
        for n, q in enumerate(queries):
            try:
                idx = get_index(q.video_id)
            except FileNotFoundError:
                continue
            _t = time.perf_counter()
            dur = float(idx.ts[-1]) + 1.0 / args.fps
            ch = chunks_for(dur)
            sims = idx.emb.astype(np.float32) @ qemb[n]
            cs = score_chunks(idx.ts, sims, ch, agg=args.agg, topk=args.topk)
            ranked = np.argsort(-cs)
            t_select += time.perf_counter() - _t

            if budget > 0 and scorer is not None:
                # re-rank the cheap stage's top-N with the expensive model: one
                # frame per chunk, the chunk's best-scoring frame
                top = ranked[:budget]
                nf = max(1, args.rerank_frames)
                pick_ts, per_chunk = [], []
                for ci in top:
                    lo = np.searchsorted(idx.ts, ch[ci][0], "left")
                    hi = np.searchsorted(idx.ts, ch[ci][1], "right")
                    if nf == 1:
                        if hi <= lo:
                            ts_here = [float(ch[ci][0])]
                        else:
                            ts_here = [float(idx.ts[lo + int(np.argmax(sims[lo:hi]))])]
                    else:
                        # spread the frames across the chunk so the model sees
                        # the whole ten seconds, not the same instant nf times
                        ts_here = list(np.linspace(ch[ci][0], max(ch[ci][0],
                                                                 ch[ci][1] - 0.2), nf))
                    per_chunk.append(len(ts_here))
                    pick_ts.extend(ts_here)
                from framesieve.fetch import FrameFetcher
                fetcher = FrameFetcher(q.video_path, workers=16)
                _t = time.perf_counter()
                _, frames = fetcher.fetch(pick_ts)
                t_fetch += time.perf_counter() - _t
                if len(frames) >= sum(per_chunk):
                    _t = time.perf_counter()
                    question = f"Does this show: {q.text}"
                    if nf == 1:
                        vs = []
                        for i in range(0, len(frames), 16):
                            vs.append(scorer.score(list(frames[i:i + 16]), question))
                        vs = np.concatenate(vs)
                    else:
                        clips, off = [], 0
                        for c in per_chunk:
                            clips.append(list(frames[off:off + c])); off += c
                        vs = []
                        for i in range(0, len(clips), 4):
                            vs.append(scorer.score_clips(clips[i:i + 4], question))
                        vs = np.concatenate(vs)
                    t_vlm += time.perf_counter() - _t
                    vlm_calls += len(frames)
                    order = np.argsort(-vs)
                    # dtype matters here: on a video with fewer chunks than the
                    # budget the "rest" list is empty, and an empty np.array is
                    # float64, which silently turns the whole index array float
                    # and blows up several hundred queries later
                    rest = np.array([c for c in ranked
                                     if c not in set(top.tolist())], dtype=np.int64)
                    ranked = np.concatenate([top[order].astype(np.int64), rest])

            is_gt = gt_chunk_mask(ch, q.gt_intervals)
            a = recall_at_k(ranked, is_gt, 1)
            b = recall_at_k(ranked, is_gt, 5)
            c = map_at_5(ranked, ch, q.gt_intervals)
            d = map_at_5_matched(ranked, ch, q.gt_intervals)
            r1.append(a); r5.append(b); m5.append(c); m5m.append(d)
            by_task.setdefault(q.meta_task, []).append((a, c))
            if budget > 0 and (n + 1) % 100 == 0:
                el = time.perf_counter() - t0
                print(f"  budget={budget}: {n+1}/{len(queries)} R@1={np.mean(r1):.3f} "
                      f"({el/(n+1)*1000:.0f} ms/q)", flush=True)

        mr1, lo1, hi1 = bootstrap(np.array(r1))
        mm5, lo2, hi2 = bootstrap(np.array(m5))
        row = dict(vlm_budget=budget, n_queries=len(r1),
                   R1=mr1 * 100, R1_lo=lo1 * 100, R1_hi=hi1 * 100,
                   R5=float(np.mean(r5)) * 100,
                   mAP5=mm5 * 100, mAP5_lo=lo2 * 100, mAP5_hi=hi2 * 100,
                   mAP5_matched=float(np.mean(m5m)) * 100,
                   vlm_calls=vlm_calls, wall_s=time.perf_counter() - t0,
                   select_s=t_select, fetch_s=t_fetch, vlm_s=t_vlm,
                   n_model_calls=budget * max(1, len(r1)) if budget else 0,
                   by_meta_task={k: dict(R1=float(np.mean([x[0] for x in v])) * 100,
                                         mAP5=float(np.mean([x[1] for x in v])) * 100,
                                         n=len(v))
                                 for k, v in sorted(by_task.items())})
        all_rows.append(row)
        tag = "retrieval only" if budget == 0 else f"cascade, {budget} VLM calls/query"
        nq = max(1, len(r1))
        print(f"[{tag}]  R@1 {row['R1']:.2f} [{lo1*100:.2f},{hi1*100:.2f}]   "
              f"mAP@5 {row['mAP5_matched']:.2f}   "
              f"per query: select {t_select/nq*1000:.1f} ms, "
              f"fetch {t_fetch/nq*1000:.0f} ms, vlm {t_vlm/nq*1000:.0f} ms", flush=True)
        with open(args.out, "w") as f:
            json.dump(dict(config=vars(args), encoder=enc.describe(),
                           chunk_s=CHUNK_S, iou_threshold=IOU_THRESHOLD,
                           paper_baselines=PAPER_BASELINES, rows=all_rows), f, indent=2)

    print(f"\n{'method':<34}{'R@1':>8}{'mAP@5':>9}")
    print("-" * 51)
    for k, (a, b) in sorted(PAPER_BASELINES.items(), key=lambda kv: kv[1][0]):
        print(f"{k + ' (paper, all modalities)':<34}{a:>8.1f}{b:>9.1f}")
    for row in all_rows:
        tag = ("framesieve retrieval only" if row["vlm_budget"] == 0
               else f"framesieve + VLM x{row['vlm_budget']}")
        print(f"{tag:<34}{row['R1']:>8.2f}{row['mAP5']:>9.2f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
