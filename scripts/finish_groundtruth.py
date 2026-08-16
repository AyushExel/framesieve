"""Drive the two ground-truth shards to completion and merge them.

Shard 1 was launched over the whole video but only its first half is wanted;
shard 2 covers [SPLIT, end]. Once shard 1's progress passes the split point,
stop it -- everything beyond would be duplicated work -- and let shard 2 have the
GPU to itself for the remainder.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time

SPLIT = 8128
# The log advances every 80 frames but the checkpoint is only written every 320,
# so the saved file lags what the log reports. Overshoot by more than one
# checkpoint interval before stopping shard 1, or the merge has a hole in it
# exactly at the seam -- which merge_groundtruth.py would catch, but only after
# wasting the run.
SPLIT_MARGIN = 400
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P1_LOG = os.path.join(ROOT, "runs/gt_build.log")
P2_LOG = os.path.join(ROOT, "runs/gt_part2.log")
_LINE = re.compile(r"([\d,]+)/[\d,]+ frames")


def frames_done(path: str) -> int:
    try:
        with open(path, "rb") as f:
            data = f.read().decode("utf8", "ignore")
    except OSError:
        return 0
    m = _LINE.findall(data)
    return int(m[-1].replace(",", "")) if m else 0


def pids(pattern: str) -> list[int]:
    out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    return [int(x) for x in out.stdout.split() if x.strip().isdigit()]


def alive(pattern: str) -> bool:
    return len(pids(pattern)) > 0


P1 = "build_groundtruth.*groundtruth_glasgow"
P2 = "build_groundtruth.*gt_part2"


def main() -> None:
    target = SPLIT + SPLIT_MARGIN
    print(f"waiting for shard 1 to pass {target} frames "
          f"(split {SPLIT} + {SPLIT_MARGIN} checkpoint margin)", flush=True)
    while True:
        n = frames_done(P1_LOG)
        if n >= target:
            print(f"shard 1 at {n:,} frames -- stopping it", flush=True)
            for p in pids(P1):
                os.kill(p, signal.SIGTERM)
            time.sleep(8)
            break
        if not alive(P1):
            print(f"shard 1 exited on its own at {n:,} frames", flush=True)
            break
        time.sleep(30)

    print("waiting for shard 2", flush=True)
    while alive(P2):
        time.sleep(30)
    print(f"shard 2 done at {frames_done(P2_LOG):,} frames", flush=True)

    py = os.path.join(ROOT, ".venv/bin/python")
    r = subprocess.run(
        [py, os.path.join(ROOT, "scripts/merge_groundtruth.py"),
         os.path.join(ROOT, "runs/groundtruth_glasgow.npz"),
         os.path.join(ROOT, "runs/gt_part2.npz"),
         "--out", os.path.join(ROOT, "runs/groundtruth_merged.npz")],
        capture_output=True, text=True)
    print(r.stdout, r.stderr, flush=True)
    if r.returncode != 0:
        sys.exit("merge failed -- shards disagree; not overwriting")
    os.replace(os.path.join(ROOT, "runs/groundtruth_merged.npz"),
               os.path.join(ROOT, "runs/groundtruth_glasgow.npz"))
    print("GROUND TRUTH COMPLETE", flush=True)


if __name__ == "__main__":
    main()
