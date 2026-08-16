"""Does shrinking the VLM's visual-token budget change its answers?

Ground truth for the whole project is a dense VLM pass, and its cost scales with
`max_pixels`. Before spending GPU-hours at native resolution, check what the
cheaper budgets actually cost in agreement. If 128 tokens tracks native closely,
ground truth gets ~2x cheaper for free; if it does not, we pay for native and say
so.

Agreement is measured three ways, because they answer different questions:
  spearman   - does the ranking survive? (this is what a recall curve depends on)
  sign agree - would a yes/no decision flip?
  AUC vs native - can the cheap budget recover the expensive budget's positives?
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from framesieve.frames import FrameStream  # noqa: E402
from framesieve.vlm import QwenYesNoScorer  # noqa: E402

QUERIES = [
    "Is the train inside a tunnel?",
    "Is there a railway station platform?",
    "Is there a lake, loch or large body of water?",
    "Is there another train visible?",
]


def sample_frames(video: str, n: int, size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Stream at 1 fps and keep a random subset, so the sample is unbiased in time."""
    rng = np.random.default_rng(seed)
    stream = FrameStream(video, target_fps=1.0, size=size, batch=256)
    keep_ts, keep_fr = [], []
    total = stream.n_expected
    want = rng.choice(total, size=min(n, total), replace=False)
    want_set = set(int(x) for x in want)
    i = 0
    for ts, fr in stream:
        for j in range(len(fr)):
            if i in want_set:
                keep_ts.append(ts[j])
                keep_fr.append(fr[j])
            i += 1
    return np.array(keep_ts), np.stack(keep_fr)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra @ rb) / (np.linalg.norm(ra) * np.linalg.norm(rb) + 1e-12))


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/glasgow_mallaig.mp4")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--budgets", type=int, nargs="*", default=[64, 128, 256])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/vlm_token_budget.json")
    args = ap.parse_args()

    ts, frames = sample_frames(args.video, args.n, size=720, seed=args.seed)
    print(f"sampled {len(frames)} frames at {frames.shape[2]}x{frames.shape[1]}\n")

    scores: dict[int, dict[str, np.ndarray]] = {}
    timing: dict[int, float] = {}
    for tb in args.budgets:
        px = tb * 28 * 28 * 4
        sc = QwenYesNoScorer(max_pixels=px, min_pixels=min(px, 64 * 28 * 28))
        import time
        t0 = time.perf_counter()
        scores[tb] = {}
        for q in QUERIES:
            out = []
            for i in range(0, len(frames), 16):
                out.append(sc.score(list(frames[i:i + 16]), q))
            scores[tb][q] = np.concatenate(out)
        timing[tb] = time.perf_counter() - t0
        print(f"budget {tb:>4} tokens: {timing[tb]:.1f} s for {len(QUERIES)} queries "
              f"x {len(frames)} frames  ({len(QUERIES)*len(frames)/timing[tb]:.1f} scores/s)")
        del sc
        torch.cuda.empty_cache()

    ref = max(args.budgets)
    print(f"\nreference budget = {ref} tokens\n")
    print(f"{'query':<46}{'budget':>8}{'spearman':>10}{'sign agr':>10}{'AUC':>8}"
          f"{'pos rate':>10}")
    print("-" * 92)
    summary = []
    for q in QUERIES:
        rs = scores[ref][q]
        lab = rs > 0
        for tb in args.budgets:
            s = scores[tb][q]
            row = dict(query=q, budget=tb, spearman=spearman(s, rs),
                       sign_agreement=float(((s > 0) == lab).mean()),
                       auc_vs_ref=auc(s, lab), pos_rate=float((s > 0).mean()))
            summary.append(row)
            print(f"{q[:44]:<46}{tb:>8}{row['spearman']:>10.3f}"
                  f"{row['sign_agreement']:>10.3f}{row['auc_vs_ref']:>8.3f}"
                  f"{row['pos_rate']:>10.3f}")
        print()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "timing_s": timing, "rows": summary,
                   "scores": {str(k): {q: v.tolist() for q, v in d.items()}
                              for k, d in scores.items()},
                   "ts": ts.tolist()}, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
