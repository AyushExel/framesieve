"""framesieve -- search long video without running a VLM on every frame.

    import framesieve as fs

    video = fs.open("holiday.mp4")                 # index once, ~15 s per hour
    for hit in video.search("a dark tunnel"):      # ~1 ms per query after that
        print(hit.timecode, hit.score)

    hits = video.search("a dark tunnel", k=8, confirm=True)   # ask a real VLM
    print(hits.above(0).to_dicts())

Indexing decodes the video at one frame per second and embeds every frame with a
small image encoder. Searching is then a matrix multiply against that index,
with the expensive vision-language model spent only on the handful of frames that
survive -- which is the whole point, because that model costs about 809x the
cheap one per frame.

`import framesieve` deliberately imports nothing heavy -- about 50 ms, and no
torch. Building an index needs torch; READING one does not, so you can query and
inspect indexes on a machine with no GPU stack at all. The submodules
(`framesieve.indexing`, `.search`, `.encoders`, `.vlm`, `.pooling`) are the
lower level and stay importable. (The indexing module is deliberately NOT named
`index`: a submodule by that name fights the `index()` function for the
attribute, and `import framesieve.index as m` used to hand back the function.)
"""

from __future__ import annotations

__version__ = "0.2.0"
try:
    from .api import (  # noqa: E402
        DEFAULT_ENCODER,
        DEFAULT_VLM,
        Hit,
        SearchResults,
        VideoIndex,
        index,
        index_path_for,
        load,
        open,
    )
except ImportError as exc:  # pragma: no cover - depends on the environment
    # A bare "No module named numpy" traceback is the most likely first
    # experience for someone running the wrong interpreter, and it does not say
    # what to do. Only the known dependencies are rewritten; a genuine
    # ImportError inside the package still surfaces as itself.
    if exc.name in {"numpy", "torch", "transformers"}:
        raise ImportError(
            f"framesieve needs {exc.name}, and this interpreter does not have "
            f"it.\n"
            f"  Install torch first, from the index for your platform:\n"
            f"    pip install torch --index-url "
            f"https://download.pytorch.org/whl/cu128\n"
            f"  then:\n"
            f"    pip install framesieve"
        ) from exc
    raise

__all__ = ["Hit", "SearchResults", "VideoIndex", "index", "load", "open",
           "index_path_for", "DEFAULT_ENCODER", "DEFAULT_VLM",
           "Collection", "CollectionHit", "pooling", "__version__"]


def __getattr__(name: str):
    """`framesieve.pooling` and `Collection` on first touch, so the score-pooling helpers are
    reachable as an attribute without importing them for everyone.

    `import_module` rather than `from . import pooling`: the latter resolves the
    name through getattr on this package, which lands back in this function and
    recurses until the stack runs out.
    """
    if name == "pooling":
        import importlib
        return importlib.import_module(".pooling", __name__)
    if name in ("Collection", "CollectionHit"):
        # deferred because it needs lancedb, which is an optional extra: a
        # single video never touches it
        import importlib
        return getattr(importlib.import_module(".collection", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
