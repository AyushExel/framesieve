"""Merge ground-truth shards produced by parallel build_groundtruth runs.

The dense VLM pass is GPU-bound but leaves the GPU around half idle waiting on
CPU-side image preprocessing, so it is worth running two shards over disjoint
time ranges. This stitches them back together.

Two things are checked rather than trusted:
  - the shards agree on the query list and on the model/settings, so scores are
    comparable
  - where the shards overlap in time, they agree on the score. They should agree
    exactly: the VLM is deterministic, seeked decoding returns bit-identical
    frames, and both shards batch in the same size. A mismatch means one of those
    three assumptions broke, and it is better to fail loudly here than to publish
    a curve built on a seam.
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+")
    ap.add_argument("--out", default="runs/groundtruth_glasgow.npz")
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()

    parts = []
    for p in args.shards:
        z = np.load(p, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        parts.append(dict(path=p, ts=z["ts"], scores=z["scores"],
                          queries=[str(q) for q in z["queries"]], meta=meta))
        print(f"{p}: {len(z['ts']):,} frames, "
              f"t {z['ts'][0]:.0f}-{z['ts'][-1]:.0f} s")

    q0 = parts[0]["queries"]
    for p in parts[1:]:
        if p["queries"] != q0:
            raise SystemExit(f"query mismatch between {parts[0]['path']} and {p['path']}")
    for k in ("model", "max_visual_tokens", "fps"):
        vals = {json.dumps(p["meta"]["config"].get(k)
                           if k in p["meta"]["config"] else p["meta"]["model"].get(k))
                for p in parts}
        if len(vals) > 1:
            raise SystemExit(f"shards disagree on {k}: {vals}")

    # overlap check
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            a, b = parts[i], parts[j]
            common, ia, ib = np.intersect1d(a["ts"], b["ts"], return_indices=True)
            if len(common) == 0:
                continue
            d = np.abs(a["scores"][ia] - b["scores"][ib])
            print(f"  overlap {a['path']} vs {b['path']}: {len(common)} frames, "
                  f"max |delta| {d.max():.6f}")
            if d.max() > args.tol:
                raise SystemExit(
                    f"shards disagree by {d.max():.4f} on {len(common)} shared frames "
                    "-- the VLM, the decode path, or the batching is not "
                    "reproducible across shards; do not merge")

    ts = np.concatenate([p["ts"] for p in parts])
    sc = np.concatenate([p["scores"] for p in parts])
    order = np.argsort(ts, kind="stable")
    ts, sc = ts[order], sc[order]
    keep = np.concatenate([[True], np.diff(ts) > 1e-6])
    ts, sc = ts[keep], sc[keep]

    gaps = np.diff(ts)
    print(f"\nmerged: {len(ts):,} frames, t {ts[0]:.0f}-{ts[-1]:.0f} s, "
          f"median step {np.median(gaps):.2f} s, max step {gaps.max():.2f} s")
    if gaps.max() > 2.5 * np.median(gaps):
        print(f"  WARNING: a gap of {gaps.max():.1f} s suggests a missing shard")

    total_gpu_s = sum(p["meta"].get("elapsed_s", 0.0) for p in parts)
    meta = dict(parts[0]["meta"])
    meta["merged_from"] = [p["path"] for p in parts]
    meta["n_frames_done"] = int(len(ts))
    meta["elapsed_s"] = total_gpu_s
    meta["wall_note"] = ("shards ran concurrently; elapsed_s is summed GPU-process "
                         "time, not wall clock")
    np.savez_compressed(args.out, ts=ts.astype(np.float32),
                        scores=sc.astype(np.float32),
                        queries=np.array(q0, dtype=object),
                        meta=json.dumps(meta), allow_pickle=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
