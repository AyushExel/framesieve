#!/usr/bin/env python3
"""Index a folder of videos once, then run several queries across all of them.

    python examples/03_batch_a_folder.py ./footage "a red car" "someone running"

This is the shape the cost model is built for: indexing is paid once per video,
so the more queries you ask, the more the cheap stage disappears into the noise.
"""
import glob
import os
import sys
import time

import framesieve as fs

folder = sys.argv[1] if len(sys.argv) > 1 else "data"
queries = sys.argv[2:] or ["a train", "a station platform"]

paths = sorted(p for ext in ("mp4", "mkv", "mov", "webm")
               for p in glob.glob(os.path.join(folder, f"*.{ext}")))
if not paths:
    sys.exit(f"no videos in {folder}")

t0 = time.perf_counter()
videos = [fs.open(p) for p in paths]
hours = sum(v.duration for v in videos) / 3600
print(f"{len(videos)} videos, {hours:.2f} h, ready in {time.perf_counter()-t0:.1f} s")

for q in queries:
    t = time.perf_counter()
    # one hit per video, then rank across videos. Similarity is comparable
    # within a query even across videos, because the query vector is the same.
    best = []
    for v in videos:
        hits = v.search(q, k=8)
        if len(hits):
            best.append((hits[0].score, os.path.basename(str(v.video)), hits[0]))
    best.sort(reverse=True, key=lambda x: x[0])
    print(f"\n{q!r}  —  {1000*(time.perf_counter()-t):.0f} ms across all of them")
    for score, name, hit in best[:5]:
        print(f"  {score:.3f}  {name:<28} {hit.timecode}")
