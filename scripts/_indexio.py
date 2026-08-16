"""Read an index in either format, for the research scripts only.

The library reads Lance and only Lance. These scripts also have to read the 843
`.npz` artifacts under `runs/` that back every measured number in the repo, and
their `--index` arguments can point at either. Rather than put that branch back
into `FrameIndex.load`, it lives here, where it is obviously a property of this
repository's history and not of the library.

`scripts/convert_indexes.py` migrates the old files if you would rather not
carry this at all.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from framesieve.index import FrameIndex  # noqa: E402

__all__ = ["read_index"]


def read_index(path: str) -> FrameIndex:
    return FrameIndex.from_npz(path) if path.endswith(".npz") else FrameIndex.load(path)
