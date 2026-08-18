#!/usr/bin/env python3
"""Convert pre-Lance .npz indexes to the current format.

    python scripts/convert_indexes.py "runs/**/*.npz"

The library reads Lance and only Lance. This walks the old artifacts and
rewrites them, so the research scripts can be pointed at the converted copies
rather than carrying a second code path through the library for the sake of
files that only exist in this repository.
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from framesieve.indexing import FrameIndex  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern", help='e.g. "runs/**/*.npz"')
    ap.add_argument("--keep", action="store_true",
                    help="leave the .npz in place (default: leave it in place)")
    ap.add_argument("--delete", action="store_true",
                    help="remove each .npz once its Lance copy verifies")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.pattern, recursive=True))
    if not paths:
        sys.exit(f"nothing matched {args.pattern!r}")
    print(f"{len(paths)} indexes")
    for i, p in enumerate(paths, 1):
        out = p[: -len(".npz")] + ".lance"
        idx = FrameIndex.from_npz(p)
        idx.save(out)
        # verify before deleting anything, because the point of these files is
        # that they are the evidence
        back = FrameIndex.load(out)
        assert back.emb.shape == idx.emb.shape, p
        assert abs(float(back.ts[-1]) - float(idx.ts[-1])) < 1e-3, p
        if args.delete:
            os.remove(p)
        if i % 50 == 0 or i == len(paths):
            print(f"  {i}/{len(paths)}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
