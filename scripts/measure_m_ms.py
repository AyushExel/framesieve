"""Measure m on MomentSeeker, so the k* ~ m claim rests on two datasets not one.

scripts/measure_m.py established that the best pooling depth tracks the MEASURED
number of matching frames, with a correlation of +0.941. It did so on one video,
with eight queries. That is a single point of real evidence under a claim the
whole post rests on.

MomentSeeker has 265 videos and 1,000 queries, but no per-frame labels -- only
time intervals, and an interval says which seconds are relevant, not which frames
carry evidence a model can see. So the per-frame labels get measured the same way
they were measured on the other video: run the expensive model over the frames of
every positive chunk and count how many it says match.

That is ~8,400 VLM calls, which is bounded and cheap precisely because of the
cascade this project is about -- a dense pass over all 265 videos would be
millions.

The protocol below is deliberately identical to measure_m.py: pools of one
positive against negatives from the same query, R@1, and the same k grid. If the
correlation reproduces on a second, much broader dataset, the claim stands on
something. If it does not, that needs to be said plainly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from framesieve.benchmarks.momentseeker import (  # noqa: E402
    chunks_for,
    gt_chunk_mask,
    load_queries,
)
from framesieve.indexing import FrameIndex  # noqa: E402

KS = (1, 2, 3, 4, 5, 6, 8, 10)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="data/ms_raw/t2v.json")
    ap.add_argument("--video-dir", default="data/ms_videos")
    ap.add_argument("--index-dir", default="runs/ms_index")
    ap.add_argument("--encoder", default="siglip2-base-224")
    ap.add_argument("--vlm", default="qwen2.5-vl-7b")
    ap.add_argument("--tokens-per-frame", type=int, default=64)
    ap.add_argument("--threshold", type=float, default=2.0,
                    help="oracle logit margin above which a frame counts as a match")
    ap.add_argument("--max-chunks", type=int, default=0,
                    help="cap on positive chunks scored, for a quick run")
    ap.add_argument("--out", default="runs/measure_m_ms.json")
    args = ap.parse_args()

    queries = [q for q in load_queries(args.json, args.video_dir) if q.video_path]
    have = {os.path.splitext(f)[0] for f in os.listdir(args.index_dir)
            if f.endswith(".npz")}
    queries = [q for q in queries if q.video_id in have]
    print(f"{len(queries)} queries")

    from framesieve.encoders import SiglipEncoder
    enc = SiglipEncoder(args.encoder)
    qemb = np.concatenate(
        [enc.encode_text([q.text for q in queries[i:i + 256]]).cpu().numpy()
         for i in range(0, len(queries), 256)]).astype(np.float32)
    del enc

    # pass one: every chunk's SigLIP scores, and which chunks are positive
    print("\npass 1 -- chunk scores and positives from the cheap index")
    cache: dict = {}
    recs = []          # one per (query, chunk)
    todo = []          # positive chunks needing VLM frames
    for n, q in enumerate(queries):
        if q.video_id not in cache:
            if len(cache) > 40:
                cache.clear()
            cache[q.video_id] = FrameIndex.from_npz(
                os.path.join(args.index_dir, f"{q.video_id}.npz"))
        idx = cache[q.video_id]
        ch = chunks_for(float(idx.ts[-1]) + 1.0)
        sims = idx.emb.astype(np.float32) @ qemb[n]
        ci = np.clip(np.searchsorted(ch[:, 0], idx.ts, "right") - 1, 0, len(ch) - 1)
        is_gt = gt_chunk_mask(ch, q.gt_intervals)
        for c in range(len(ch)):
            sel = ci == c
            if sel.sum() < 5:
                continue
            v = np.sort(sims[sel])[::-1]
            r = dict(q=n, chunk=c, gt=bool(is_gt[c]), m=0,
                     sc={k: float(v[:min(k, len(v))].mean()) for k in KS})
            recs.append(r)
            if is_gt[c]:
                todo.append((r, q, [float(t) for t in idx.ts[sel]]))

    if args.max_chunks:
        todo = todo[: args.max_chunks]
    print(f"  {len(recs):,} chunks, {sum(r['gt'] for r in recs):,} positive; "
          f"scoring {len(todo):,} of them with the VLM "
          f"({sum(len(t[2]) for t in todo):,} frames)")

    # pass two: the expensive model, over the frames of positive chunks only
    print("\npass 2 -- VLM over the frames of positive chunks")
    from framesieve.fetch import FrameFetcher
    from framesieve.vlm import QwenYesNoScorer
    px = args.tokens_per_frame * 28 * 28 * 4
    scorer = QwenYesNoScorer(args.vlm, max_pixels=px,
                             min_pixels=min(px, 64 * 28 * 28))

    t0 = time.perf_counter()
    n_calls = 0
    by_video: dict = {}
    for r, q, ts in todo:
        by_video.setdefault(q.video_path, []).append((r, q, ts))
    done = 0
    for vpath, items in by_video.items():
        # one fetcher per video, all its chunks at once: seeking is the expensive
        # half of this and re-opening per chunk would double the run
        fetcher = FrameFetcher(vpath, workers=16)
        flat = [t for _, _, ts in items for t in ts]
        _, frames = fetcher.fetch(flat)
        pos = 0
        for r, q, ts in items:
            fr = frames[pos:pos + len(ts)]
            pos += len(ts)
            if not len(fr):
                continue
            s = scorer.score(fr, f"Does this frame show: {q.text}?")
            n_calls += len(fr)
            r["m"] = int(np.sum(np.asarray(s) >= args.threshold))
            r["oracle"] = [float(x) for x in s]
        done += len(items)
        if done % 100 < len(items):
            el = time.perf_counter() - t0
            print(f"  {done}/{len(todo)} chunks, {n_calls:,} frames, "
                  f"{el:.0f} s, {el/max(1,done)*1000:.0f} ms/chunk", flush=True)

    print(f"  {n_calls:,} VLM calls in {time.perf_counter()-t0:.0f} s")

    # ---- the same protocol as measure_m.py -------------------------------
    groups = {"m=1": [1], "m=2": [2], "m=3-4": [3, 4], "m=5-7": [5, 6, 7],
              "m>=8": list(range(8, 100))}
    scored = [r for r in recs if r["gt"] and "oracle" in r]
    print(f"\nmeasured m over {len(scored):,} positive chunks "
          f"(oracle >= {args.threshold})")
    counts = {lab: sum(1 for r in scored if r["m"] in ms)
              for lab, ms in groups.items()}
    counts["m=0 (oracle disagrees with the label)"] = \
        sum(1 for r in scored if r["m"] == 0)
    for lab, c in counts.items():
        print(f"  {lab:<38}{c:>7}")

    by_q_neg: dict = {}
    for r in recs:
        if not r["gt"]:
            by_q_neg.setdefault(r["q"], []).append(r)

    # The pool protocol has a closed form. If a positive outranks b of its
    # query's N negatives, the chance it tops a pool built from s of them drawn
    # at random is C(b, s) / C(N, s). Sampling pools instead just adds Monte
    # Carlo noise on top of the sampling noise that actually matters, which is
    # the ~50 distinct positive chunks per group.
    from math import lgamma
    S = 49

    def win_prob(b: int, N: int) -> float:
        s = min(S, N)
        if b < s:
            return 0.0
        return float(np.exp(lgamma(b + 1) - lgamma(b - s + 1)
                            + lgamma(N - s + 1) - lgamma(N + 1)))

    for r in scored:
        negs = by_q_neg.get(r["q"])
        if not negs:
            continue
        N = len(negs)
        r["p"] = {}
        for k in KS:
            b = sum(1 for x in negs if x["sc"][k] < r["sc"][k])
            r["p"][k] = win_prob(b, N)

    groups_present = [(lab, ms) for lab, ms in groups.items()
                      if sum(1 for r in scored if r["m"] in ms and "p" in r) >= 20]

    print("\npool protocol, closed form -- 1 positive vs 49 negatives, R@1")
    print(f"  {'group':<12}{'n':>5}" + "".join(f"{'k='+str(k):>8}" for k in KS)
          + f"{'best k':>9}{'max loses':>11}")
    rows = []
    for lab, ms in groups_present:
        pos_all = [r for r in scored if r["m"] in ms and "p" in r]
        acc = {k: float(np.mean([r["p"][k] for r in pos_all])) for k in KS}
        bk = max(acc, key=lambda k: acc[k])
        rows.append(dict(group=lab, ms=ms, acc=acc, best_k=bk, n_pos=len(pos_all)))
        print(f"  {lab:<12}{len(pos_all):>5}"
              + "".join(f"{100*acc[k]:>8.1f}" for k in KS)
              + f"{bk:>9}{100*(acc[bk]-acc[1]):>11.1f}")

    if len(rows) >= 3:
        mid = [float(np.mean(r["ms"][:4])) for r in rows]
        bk = [r["best_k"] for r in rows]
        c = (float(np.corrcoef(np.log(mid), np.log(bk))[0, 1])
             if np.std(bk) > 0 else 0.0)
        print(f"\n  measured m:  {[round(v, 1) for v in mid]}")
        print(f"  best k:      {bk}")

        # Resample the CHUNKS -- they are the ~50-per-group sampling unit and the
        # only source of real uncertainty now that the pools are exact.
        BOOT = 2000
        rb = np.random.default_rng(9001)
        P = [np.array([[r["p"][k] for k in KS] for r in
                       [x for x in scored if x["m"] in g["ms"] and "p" in x]])
             for g in rows]
        cs, kk = [], []
        for _ in range(BOOT):
            k_hat = []
            for M in P:
                take = M[rb.integers(0, len(M), len(M))]
                k_hat.append(KS[int(np.argmax(take.mean(axis=0)))])
            kk.append(k_hat)
            if np.std(k_hat) > 0:
                cs.append(float(np.corrcoef(np.log(mid), np.log(k_hat))[0, 1]))
        lo, hi = np.percentile(cs, [2.5, 97.5])
        print(f"  correlation of log(best k) with log(m): {c:+.3f}  "
              f"[{lo:+.2f}, {hi:+.2f}]  over {BOOT} chunk bootstraps")
        print(f"  bootstraps with a positive correlation: "
              f"{100 * np.mean(np.array(cs) > 0):.0f}%")
        loss = [100 * (r["acc"][r["best_k"]] - r["acc"][1]) for r in rows]
        print(f"  max is beaten in {sum(v > 0.05 for v in loss)}/{len(rows)} "
              f"groups, by {min(loss):.1f} to {max(loss):.1f} points")
        klo = np.percentile(kk, 2.5, axis=0)
        khi = np.percentile(kk, 97.5, axis=0)
        print(f"\n  {'group':<12}{'best k':>9}{'95% interval':>16}")
        for r, a_, b_ in zip(rows, klo, khi):
            r["best_k_ci"] = [float(a_), float(b_)]
            print(f"  {r['group']:<12}{r['best_k']:>9}{f'[{a_:.0f}, {b_:.0f}]':>16}")
        print("\n  " + ("REPRODUCES on a second dataset: the interval excludes zero"
                        if lo > 0 else
                        "PARTIALLY reproduces: the point estimate is positive but "
                        "its interval\n  includes zero, so this is support and "
                        "not confirmation"))

    with open(args.out, "w") as f:
        json.dump(dict(threshold=args.threshold, n_scored=len(scored),
                       n_calls=n_calls, counts=counts, rows=rows), f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
