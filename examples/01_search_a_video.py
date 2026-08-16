#!/usr/bin/env python3
"""Index a video and search it — the whole library in twenty lines.

    python examples/01_search_a_video.py my_video.mp4 "a dark tunnel"

The first run indexes (about 15 s per hour of video) and writes a sidecar next
to the file. Every run after that reads the sidecar, and the search itself is a
matrix multiply.
"""
import sys

import framesieve as fs

video_path = sys.argv[1] if len(sys.argv) > 1 else "data/demo_clip.mp4"
query = sys.argv[2] if len(sys.argv) > 2 else "a train"

video = fs.open(video_path)
print(video)

print(f"\nretrieval only — {len(video):,} frames ranked:")
for hit in video.search(query, k=5):
    print(f"  {hit.timecode}   similarity {hit.score:.3f}")

# confirm=True fetches those frames and asks a vision-language model whether they
# actually show what you asked for. About 30 ms a frame, and the difference
# between "looks similar" and "is".
print("\nwith a model actually looking:")
hits = video.search(query, k=5, confirm=True)
for hit in hits:
    verdict = "yes" if hit.confirmed else "no "
    print(f"  {hit.timecode}   {verdict}  log-odds {hit.vlm_score:+.2f}")

kept = hits.above(0.0)
print(f"\n{len(kept)} of {len(hits)} confirmed, in {hits.latency_ms:.0f} ms")
