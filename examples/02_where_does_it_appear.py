#!/usr/bin/env python3
"""Plot where in a video a concept appears, using the whole score curve.

    python examples/02_where_does_it_appear.py my_video.mp4 "a tunnel" out.png

`search()` gives you the top k. `score()` gives you every frame, which is what
you want for a timeline, a heatmap, or finding every run above some level rather
than the single best moment.
"""
import sys

import numpy as np

import framesieve as fs

video_path = sys.argv[1] if len(sys.argv) > 1 else "data/demo_clip.mp4"
query = sys.argv[2] if len(sys.argv) > 2 else "a train"
out = sys.argv[3] if len(sys.argv) > 3 else "where.png"

video = fs.open(video_path)
curve = video.score(query)          # one number per indexed frame
times = video.times

# Similarity has no absolute scale, so "high" has to be defined relative to this
# video: a fixed threshold like 0.2 would mean something different on every clip.
hot = curve > np.percentile(curve, 98)
runs = np.split(times[hot], np.flatnonzero(np.diff(np.flatnonzero(hot)) > 3) + 1)
print(f"{len(runs)} stretch(es) in the top 2% for {query!r}:")
for r in runs[:10]:
    if len(r):
        print(f"  {r[0]:8.1f} s to {r[-1]:8.1f} s   ({len(r)} frames)")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("\n(install matplotlib to also write the plot)")

fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(times / 60, curve, linewidth=1)
ax.fill_between(times / 60, curve.min(), curve, where=hot, alpha=0.3)
ax.set_xlabel("minutes")
ax.set_ylabel("similarity")
ax.set_title(f"{query!r} across {video.duration/60:.0f} minutes")
fig.tight_layout()
fig.savefig(out, dpi=150)
print(f"\nwrote {out}")
