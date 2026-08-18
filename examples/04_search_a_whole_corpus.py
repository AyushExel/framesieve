#!/usr/bin/env python3
"""Search across many videos at once, from disk rather than from RAM.

    python examples/04_search_a_whole_corpus.py ./footage "a red car"

One video fits in memory: an hour at 1 fps is 3,600 vectors. A corpus does not
-- 10,000 hours is 36 million vectors and 110 GB -- so past a few hundred hours
the vectors belong on disk with a vector index over them.

`Collection` is that: LanceDB underneath, one row per frame, and a search that
returns which video as well as when.

Indexing is the expensive half and it is per-video, so it parallelises: index
each video wherever you like, then merge the sidecars here with add_indexes().
"""
import glob
import os
import sys
import time

import framesieve as fs

folder = sys.argv[1] if len(sys.argv) > 1 else "data"
queries = sys.argv[2:] or ["a person talking", "a car", "a computer screen"]
uri = os.path.join(folder, "_framesieve.lancedb")

lib = fs.Collection(uri)

if len(lib) == 0:
    # existing indexes first, since merging them skips re-encoding entirely;
    # speech/OCR sidecars and already-added videos are skipped automatically
    merged = lib.add_indexes(os.path.join(folder, "*.lance"))
    if not merged:
        vids = sorted(p for ext in ("mp4", "mkv", "mov", "webm")
                      for p in glob.glob(os.path.join(folder, f"*.{ext}")))
        if not vids:
            sys.exit(f"no videos or indexes in {folder}")
        print(f"indexing {len(vids)} videos")
        for i, v in enumerate(vids, 1):
            n = lib.add(v)
            print(f"  {i}/{len(vids)}  {os.path.basename(v)}  {n:,} frames")

    print(f"\nbuilding the vector index over {len(lib):,} frames")
    t = time.perf_counter()
    lib.build_ann()
    print(f"  {time.perf_counter() - t:.0f} s")

print(f"\n{lib}\n")

for q in queries:
    t = time.perf_counter()
    hits = lib.search(q, k=5)
    dt = (time.perf_counter() - t) * 1000
    print(f'"{q}"  —  {dt:.1f} ms across the whole collection')
    for h in hits:
        print(f"   {h.score:.3f}  {os.path.basename(h.video):<28} {h.timecode}")
    print()

# An approximate index trades recall for speed. Check what yours costs rather
# than assuming: it depends on how your embeddings are distributed, not on the
# library.
print(f"recall@5 against an exact scan: {100 * lib.recall_at(queries, k=5):.0f}%")
