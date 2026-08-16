"""Checks that the numbers in this repo mean what they claim to mean.

Run this before believing anything. It is deliberately opinionated about *which*
assumptions get checked: the ones that, if wrong, would silently invalidate a
headline result rather than crash.

  1. env          -- records platform, driver, versions, model revisions
  2. nvdec        -- the GPU decode path actually loads (this host shipped without
                     libnvcuvid, and every NVDEC number was a silent failure until
                     it was installed)
  3. decode grid  -- FrameStream and FrameFetcher return the *same* frame for the
                     same timestamp. The recall curve looks up ground-truth scores
                     instead of re-running the VLM; that is only valid if the two
                     frame paths agree.
  4. vlm lookup   -- re-scoring fetched frames with the VLM reproduces the stored
                     ground-truth scores. This is check 3 carried all the way
                     through the expensive model.
  5. determinism  -- the encoder and the selection strategies give bit-identical
                     results on a repeat run with the same seed.
  6. index sanity -- embeddings are unit-norm, timestamps monotone, segments
                     partition the frames.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from framesieve.encoders import SIGLIP_MODELS, SiglipEncoder  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _indexio import read_index  # noqa: E402
from framesieve.fetch import FrameFetcher  # noqa: E402
from framesieve.frames import FrameStream  # noqa: E402
from framesieve.index import FrameIndex  # noqa: E402
from framesieve.search import select_candidates  # noqa: E402
from framesieve.vlm import QWEN_MODELS  # noqa: E402

OK, FAIL, WARN = "  ok  ", " FAIL ", " warn "
_results: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> bool:
    _results.append((name, passed, detail))
    print(f"[{OK if passed else FAIL}] {name}" + (f"  --  {detail}" if detail else ""),
          flush=True)
    return passed


def note(name: str, detail: str) -> None:
    print(f"[{WARN}] {name}  --  {detail}", flush=True)


def _sh(c: str) -> str:
    try:
        return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        return "?"


# --------------------------------------------------------------------------


def check_env() -> dict:
    env = {
        "platform": platform.platform(), "machine": platform.machine(),
        "python": sys.version.split()[0], "torch": torch.__version__,
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "gpu": _sh("nvidia-smi --query-gpu=name,memory.total,driver_version "
                   "--format=csv,noheader"),
        "cpu_count": os.cpu_count(),
        "ffmpeg": _sh("ffmpeg -version 2>/dev/null | head -1"),
        "git_commit": _sh("git rev-parse HEAD"),
        "git_dirty": bool(_sh("git status --porcelain")),
        "model_revisions": {
            **{k: v["revision"] for k, v in SIGLIP_MODELS.items()},
            **{k: v["revision"] for k, v in QWEN_MODELS.items()}},
    }
    print(json.dumps(env, indent=2))
    check("cuda available", torch.cuda.is_available())
    if env["git_dirty"]:
        note("git worktree", "dirty -- numbers may not match this commit")
    return env


def check_nvdec() -> None:
    n = _sh("ldconfig -p | grep -c nvcuvid")
    check("libnvcuvid present", n not in ("", "0"),
          "install libnvidia-decode-<driver> if this fails; NVDEC silently "
          "falls back otherwise")
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-v", "error", "-f", "lavfi", "-i",
         "testsrc=size=320x240:rate=25:duration=2", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-f", "mp4", "-y", "/tmp/_fs_verify.mp4"],
        capture_output=True)
    p = subprocess.run(
        ["ffmpeg", "-hide_banner", "-v", "error", "-c:v", "h264_cuvid",
         "-i", "/tmp/_fs_verify.mp4", "-f", "null", "-"], capture_output=True)
    err = p.stderr.decode("utf8", "ignore")
    check("nvdec decodes", p.returncode == 0 and "Cannot load" not in err,
          err.strip().splitlines()[-1][:100] if p.returncode else "")


def check_decode_grid(video: str, n: int = 12) -> None:
    """FrameStream(t) and FrameFetcher(t) must agree, or ground-truth lookup lies."""
    stream = FrameStream(video, target_fps=1.0, size=None, batch=64, duration_s=120)
    ts_all, fr_all = [], []
    for ts, fr in stream:
        ts_all.append(ts); fr_all.append(fr)
        if sum(len(x) for x in ts_all) >= 120:
            break
    ts_all = np.concatenate(ts_all); fr_all = np.concatenate(fr_all)

    rng = np.random.default_rng(0)
    pick = rng.choice(len(ts_all), size=min(n, len(ts_all)), replace=False)
    fetcher = FrameFetcher(video, workers=8)
    got_ts, got_fr = fetcher.fetch(ts_all[pick].tolist())

    if len(got_fr) != len(pick):
        check("fetch returns every requested frame", False,
              f"{len(got_fr)}/{len(pick)}")
        return
    diffs = np.array([np.abs(fr_all[p].astype(np.int16) - g.astype(np.int16)).mean()
                      for p, g in zip(pick, got_fr)])
    ident = float((diffs < 0.5).mean())
    check("stream and seek return the same frame",
          ident >= 0.95,
          f"{ident*100:.0f}% identical, mean abs diff {diffs.mean():.3f} grey levels "
          f"(max {diffs.max():.2f})")


def check_vlm_lookup(gt_path: str, video: str, n: int = 12) -> None:
    """Re-score stored ground-truth frames; scores must reproduce."""
    if not os.path.exists(gt_path):
        note("vlm ground-truth lookup", f"{gt_path} missing, skipped")
        return
    from framesieve.vlm import QwenYesNoScorer

    z = np.load(gt_path, allow_pickle=True)
    gt_ts, gt_sc = z["ts"], z["scores"]
    queries = [str(q) for q in z["queries"]]
    meta = json.loads(str(z["meta"]))
    tb = meta["config"]["max_visual_tokens"]

    rng = np.random.default_rng(0)
    pick = rng.choice(len(gt_ts), size=min(n, len(gt_ts)), replace=False)
    fetcher = FrameFetcher(video, workers=8)
    got_ts, frames = fetcher.fetch(gt_ts[pick].tolist())

    px = tb * 28 * 28 * 4
    sc = QwenYesNoScorer(meta["config"]["model"], max_pixels=px,
                         min_pixels=min(px, 64 * 28 * 28))
    qi = 0
    redo = sc.score(list(frames), queries[qi])
    ref = gt_sc[pick, qi]
    d = np.abs(redo - ref)
    check("vlm scores reproduce from disk", float(d.max()) < 0.5,
          f"max |delta| {d.max():.4f}, mean {d.mean():.4f} on query {queries[qi]!r}")
    del sc
    torch.cuda.empty_cache()


def check_determinism(video: str) -> None:
    from framesieve.index import build_index
    enc = SiglipEncoder("siglip2-base-224")
    a = build_index(video, enc, target_fps=1.0, duration_s=60, segment_tau=0.90,
                    seed=0, verbose=False)
    b = build_index(video, enc, target_fps=1.0, duration_s=60, segment_tau=0.90,
                    seed=0, verbose=False)
    check("index build is deterministic",
          np.array_equal(a.emb, b.emb) and np.array_equal(a.seg_id, b.seg_id),
          f"emb equal={np.array_equal(a.emb, b.emb)}, "
          f"seg equal={np.array_equal(a.seg_id, b.seg_id)}")

    q = enc.encode_text(["a stone bridge"]).cpu().numpy()[0].astype(np.float32)
    for strat in ("topk", "nms", "segment"):
        c1 = select_candidates(a, q, 8, strategy=strat, seed=0)
        c2 = select_candidates(a, q, 8, strategy=strat, seed=0)
        check(f"selection deterministic: {strat}", np.array_equal(c1.ts, c2.ts))
    u1 = select_candidates(a, q, 8, strategy="uniform", seed=0)
    u2 = select_candidates(a, q, 8, strategy="uniform", seed=0)
    u3 = select_candidates(a, q, 8, strategy="uniform", seed=1)
    check("uniform is seed-reproducible and seed-sensitive",
          np.array_equal(u1.ts, u2.ts) and not np.array_equal(u1.ts, u3.ts))
    del enc
    torch.cuda.empty_cache()


def check_index_sanity(index_path: str) -> None:
    if not os.path.exists(index_path):
        note("index sanity", f"{index_path} missing, skipped")
        return
    idx = read_index(index_path)
    norms = np.linalg.norm(idx.emb.astype(np.float32), axis=1)
    check("embeddings unit-norm", bool(np.abs(norms - 1).max() < 2e-2),
          f"max deviation {np.abs(norms-1).max():.4f}")
    check("timestamps strictly increasing", bool(np.all(np.diff(idx.ts) > 0)))
    starts, ends, _, _ = idx.segments()
    check("segments partition the frames",
          int(ends[-1]) == len(idx.ts) and int(starts[0]) == 0
          and bool(np.all(starts[1:] == ends[:-1])),
          f"{len(starts):,} segments over {len(idx.ts):,} frames")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/glasgow_mallaig.mp4")
    ap.add_argument("--index", default="runs/index_glasgow_siglip2b224.npz")
    ap.add_argument("--gt", default="runs/groundtruth_glasgow.npz")
    ap.add_argument("--skip-vlm", action="store_true")
    ap.add_argument("--out", default="runs/verify.json")
    args = ap.parse_args()

    t0 = time.perf_counter()
    print("=" * 72); print("environment"); print("=" * 72)
    env = check_env()
    print("\n" + "=" * 72); print("gpu decode"); print("=" * 72)
    check_nvdec()
    if os.path.exists(args.video):
        print("\n" + "=" * 72); print("frame paths agree"); print("=" * 72)
        check_decode_grid(args.video)
        print("\n" + "=" * 72); print("determinism"); print("=" * 72)
        check_determinism(args.video)
        if not args.skip_vlm:
            print("\n" + "=" * 72); print("vlm ground-truth lookup"); print("=" * 72)
            check_vlm_lookup(args.gt, args.video)
    else:
        note("video checks", f"{args.video} missing, skipped")
    print("\n" + "=" * 72); print("index sanity"); print("=" * 72)
    check_index_sanity(args.index)

    n_pass = sum(1 for _, p, _ in _results if p)
    print("\n" + "=" * 72)
    print(f"{n_pass}/{len(_results)} checks passed in {time.perf_counter()-t0:.1f} s")
    for name, p, d in _results:
        if not p:
            print(f"  FAILED: {name}  {d}")
    with open(args.out, "w") as f:
        json.dump({"env": env, "checks": [{"name": n, "passed": p, "detail": d}
                                          for n, p, d in _results]}, f, indent=2)
    print(f"wrote {args.out}")
    sys.exit(0 if n_pass == len(_results) else 1)


if __name__ == "__main__":
    main()
