"""Search across many videos, without holding them all in memory.

`VideoIndex` is one video and lives in RAM: a few thousand vectors, ranked by a
matrix multiply. That is the right shape until it is not. At 1 fps a video-hour
is 3,600 vectors, so:

        100 h    360,000 vectors    1.1 GB    fine
        500 h    1.8M vectors       5.5 GB    uncomfortable
     10,000 h    36M vectors        110 GB    no

A `Collection` is the same idea backed by LanceDB instead of a numpy array. The
vectors live on disk and the search is an approximate-nearest-neighbour lookup
rather than a scan. It also answers a question a per-video index cannot: *which*
video, out of thousands.

Measured on 10,000,000 vectors -- 2,778 video-hours at 1 fps, 30.8 GB of vectors
plus a 31.8 GB HNSW index:

    open the collection        0.18 GB resident
    search, 30 queries         112 ms median, 139 ms p90
    peak resident              5.46 GB
    hard memory cap that works 8 GB      (4 GB and 2 GB are OOM-killed)
    same vectors in numpy      31 GB resident, always

So it is about 6x less memory than holding the corpus, not constant memory:
graph traversal touches a real working set. A 2,778-hour corpus searches on an
8 GB machine, and would not fit at all in numpy on a 16 GB one.

(Those vectors are real SigLIP embeddings tiled with jitter to reach the size.
That measures the index, not retrieval quality, and no recall number is quoted
from it -- the recall figures below come from the 205-hour corpus, which is
205 hours of distinct video.)

    import framesieve as fs

    lib = fs.Collection("footage.lancedb")
    lib.add("cam1.mp4")                       # index and append
    lib.add("cam2.mp4")
    lib.build_ann()                           # once, after the bulk load

    for hit in lib.search("a red car", k=20):
        print(hit.video, hit.timecode, hit.score)

Needs `pip install "framesieve[collection]"`.

One thing worth knowing before you use it: an approximate index trades recall for
speed, and this cascade already spends its budget on the top of the ranking. Use
`exact=True` to check what the approximation costs you on your own data --
`Collection.recall_at` does that comparison directly.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np

from .api import DEFAULT_ENCODER, timecode

__all__ = ["Collection", "CollectionHit"]

TABLE = "frames"


class DuplicateVideo(ValueError):
    """The video is already in the collection under this name."""


def _sql_str(s: str) -> str:
    """A string literal for a lancedb filter. Single quotes double, per SQL --
    without this a video called `Tim's dashcam.mp4` breaks every filter."""
    return "'" + s.replace("'", "''") + "'"


def _require_lancedb():
    try:
        import lancedb
    except ImportError as exc:  # pragma: no cover - environment
        raise ImportError(
            "Collection needs lancedb: pip install \"framesieve[collection]\"\n"
            "  A single video does not: framesieve.open(video) holds its "
            "vectors in memory and needs nothing extra."
        ) from exc
    return lancedb


@dataclass(frozen=True)
class CollectionHit:
    """One moment, in one video, out of the whole collection."""

    video: str
    time: float
    score: float

    @property
    def timecode(self) -> str:
        return timecode(self.time)

    def __repr__(self) -> str:
        return (f"<CollectionHit {os.path.basename(self.video)} "
                f"{self.timecode} score={self.score:.3f}>")


class Collection:
    """Many videos in one on-disk index.

    Not a drop-in for `VideoIndex`: it answers "where in the corpus", and
    deliberately does not carry the frame-selection strategies, which are a
    within-one-video concern.
    """

    def __init__(self, uri: str, *, encoder: str = DEFAULT_ENCODER,
                 device: str | None = None):
        lancedb = _require_lancedb()
        self.uri = uri
        self._encoder_name = encoder
        self._device = device
        self._db = lancedb.connect(uri)
        self._enc = None
        self._tbl = self._db.open_table(TABLE) if self._has_table() else None

    def _has_table(self) -> bool:
        """Does this database already hold our table?

        lancedb 0.37 deprecates table_names() in favour of list_tables(), which
        returns a ListTablesResponse rather than a list -- so `TABLE in names`
        against it is quietly always False, and reopening an existing collection
        reports it as empty. Unwrap the response, and fall back for older
        versions that only have table_names().
        """
        listed = getattr(self._db, "list_tables", None)
        names = listed() if listed is not None else self._db.table_names()
        names = getattr(names, "tables", names)
        return TABLE in list(names)

    # -- properties --------------------------------------------------------

    @property
    def table(self):
        if self._tbl is None:
            raise RuntimeError(
                f"{self.uri} is empty. Add a video with .add(path) or an "
                f"existing index with .add_index(path).")
        return self._tbl

    def __len__(self) -> int:
        return 0 if self._tbl is None else self._tbl.count_rows()

    def videos(self) -> list[str]:
        if self._tbl is None:
            return []
        col = self._tbl.to_lance().to_table(columns=["video"]).column("video")
        return sorted(set(col.to_pylist()))

    def __repr__(self) -> str:
        return (f"<Collection {self.uri} {len(self):,} frames from "
                f"{len(self.videos())} videos>")

    # -- building ----------------------------------------------------------

    def _append(self, video: str, ts: np.ndarray, emb: np.ndarray) -> int:
        import pyarrow as pa

        emb = np.ascontiguousarray(emb.astype(np.float32))
        dim = emb.shape[1]
        if self._tbl is not None:
            have = self._tbl.schema.field("vector").type.list_size
            if have != dim:
                # mixing dimensions would crash; mixing ENCODERS of the same
                # dimension would silently corrupt the ranking, and add_index
                # checks for that by name before it gets here
                raise ValueError(
                    f"this collection holds {have}-dimensional vectors and "
                    f"{video} brings {dim}-dimensional ones; a collection must "
                    f"be built with one encoder throughout")
            if self._tbl.count_rows(f"video = {_sql_str(video)}"):
                raise DuplicateVideo(
                    f"{video} is already in the collection; pass a different "
                    f"name via video= to add it twice deliberately")
        tbl = pa.table({
            "video": pa.array([video] * len(ts)),
            "ts": pa.array(np.asarray(ts, dtype=np.float32)),
            "vector": pa.FixedSizeListArray.from_arrays(
                pa.array(emb.reshape(-1)), dim),
        })
        if self._tbl is None:
            self._tbl = self._db.create_table(TABLE, tbl)
        else:
            self._tbl.add(tbl)
        return len(ts)

    @staticmethod
    def _sidecar_kind(index_path: str) -> str:
        """"frames" for a frame index, else what the sidecar says it is.

        Speech and OCR sidecars are also `.lance` and match the same globs; the
        two are told apart by their metadata, not their names.
        """
        import json as _json

        meta = os.path.join(index_path, "framesieve.json")
        if not os.path.exists(meta):
            return "unknown"
        with open(meta) as f:
            side = _json.load(f)
        return "frames" if "stats" in side else side.get("kind", "unknown")

    def add_index(self, index_path: str, video: str | None = None) -> int:
        """Append a sidecar that already exists, without re-encoding anything.

        This is how a corpus gets loaded: indexing is the expensive half and it
        is per-video, so it parallelises across machines and the results merge
        here.
        """
        from .indexing import FrameIndex

        if index_path.endswith(".npz"):
            raise ValueError(
                f"{index_path} is the pre-Lance index format, which the library "
                f"no longer reads. Convert it first:\n"
                f"  python scripts/convert_indexes.py '<dir>/*.npz'")
        kind = self._sidecar_kind(index_path)
        if kind != "frames":
            raise ValueError(
                f"{index_path} is a {kind} sidecar (speech or on-screen text), "
                f"not a frame index; it belongs beside its video and cannot "
                f"join a collection")
        idx = FrameIndex.load(index_path)
        enc = idx.stats.encoder
        if enc and enc != self._encoder_name:
            raise ValueError(
                f"{index_path} was built with encoder {enc!r}, but this "
                f"collection encodes queries with {self._encoder_name!r}; "
                f"mixing them silently corrupts the ranking. Open it as "
                f"Collection(uri, encoder={enc!r}) instead.")
        name = video or idx.stats.video or index_path
        return self._append(name, idx.ts, idx.emb)

    def add_indexes(self, pattern: str, verbose: bool = True) -> int:
        """Append every frame sidecar matching a glob. Returns frames added.

        Speech and OCR sidecars that match the glob are skipped with a note,
        because `*.lance` matches them too; a video already in the collection is
        skipped the same way, so a bulk load can be re-run. An encoder or
        dimension mismatch still raises -- that is a configuration error, not a
        file to skip.
        """
        paths = sorted(glob.glob(pattern))
        total = 0
        for i, p in enumerate(paths, 1):
            if self._sidecar_kind(p) != "frames":
                if verbose:
                    print(f"  skipping {os.path.basename(p)}: "
                          f"speech/OCR sidecar, not a frame index", flush=True)
                continue
            try:
                total += self.add_index(p)
            except DuplicateVideo:
                if verbose:
                    print(f"  skipping {os.path.basename(p)}: already in the "
                          f"collection", flush=True)
                continue
            if verbose and (i % 25 == 0 or i == len(paths)):
                print(f"  {i}/{len(paths)} videos, {total:,} frames", flush=True)
        return total

    def add(self, video: str, **kwargs) -> int:
        """Index a video and append it. Needs torch; `add_index` does not."""
        from .api import index as build

        vi = build(video, encoder=self._encoder_name, device=self._device,
                   save=False, **kwargs)
        return self._append(video, vi.times, vi.embeddings)

    # Measured on 739,739 SigLIP frame vectors (205 h of video), recall@20
    # against an exact scan, on one GH200 host:
    #
    #   index          build   size    recall@20   latency
    #   IvfHnswFlat      12s   2.3G       89.2%      5.2 ms   <- default
    #   IvfHnswSq        13s   635M       81.7%      5.0 ms
    #   IvfFlat          23s   2.3G       94.2%     44.8 ms
    #   IvfSq            27s   575M       80.8%     13.7 ms
    #   IvfRq            21s    86M       24.2%      6.7 ms
    #   IvfPq            28s    78M        0.0%     12.7 ms
    #   exact scan         -      -        100%      133 ms
    #
    # The quantized indexes fail, and badly. These embeddings sit in a narrow
    # band -- the best similarity in 205 hours is 0.16 and neighbours differ in
    # the third decimal -- so product and RaBitQ quantization error swamps the
    # signal being ranked. That is a property of the embeddings, not of LanceDB,
    # and it is why the default here is an unquantized graph rather than the
    # usual IVF_PQ advice.
    #
    # recall@20 is the wrong headline, though, and it ranks these backwards.
    # What a caller acts on is the top hit after runs are collapsed, and on that
    # measure -- 15 queries, does the top moment match an exact scan --
    # IvfHnswFlat beats IvfFlat outright:
    #
    #   nprobes         20    50   100   200   400
    #   IvfFlat        6/15  9/15 11/15 12/15 15/15   at 81 ms
    #   IvfHnswFlat   14/15 15/15 15/15 15/15 15/15   at 10 ms, flat
    #
    # IVF only matches the graph by probing half its partitions, for 8x the
    # latency, because partition scanning grows with nprobes and graph traversal
    # does not. Hence nprobes defaults to 50: one more correct answer than 20,
    # at the same cost.
    INDEXES = {
        "hnsw": "IvfHnswFlat",     # best recall per millisecond
        "hnsw_sq": "IvfHnswSq",    # a quarter of the disk, ~7 points of recall
        "flat": "IvfFlat",         # highest recall, slowest
        "sq": "IvfSq",
    }

    def build_ann(self, kind: str = "hnsw", *, num_partitions: int | None = None,
                  metric: str = "cosine", **kwargs) -> None:
        """Build the vector index. Do this once, after the bulk load.

        Without it every search scans the whole table, which is correct and gets
        linearly slower as the corpus grows.

        `kind` is one of INDEXES above. The default trades about 11 points of
        recall for a 25x speedup; pass "flat" if you would rather have the
        recall, and check what you are paying with `recall_at`.
        """
        import lancedb.index as ix

        if kind not in self.INDEXES:
            raise ValueError(f"unknown index {kind!r}; have {list(self.INDEXES)}")
        n = len(self)
        if num_partitions is None:
            # graph indexes want far fewer, larger partitions than IVF does
            num_partitions = (max(1, min(256, int(np.sqrt(max(n, 1)) / 12)))
                              if kind.startswith("hnsw")
                              else max(1, min(4096, int(np.sqrt(max(n, 1))))))
        cfg = getattr(ix, self.INDEXES[kind])(
            distance_type=metric, num_partitions=num_partitions, **kwargs)
        self.table.create_index("vector", config=cfg, replace=True)

    # -- searching ---------------------------------------------------------

    def _query_vector(self, query: str | np.ndarray) -> np.ndarray:
        if not isinstance(query, str):
            v = np.asarray(query, dtype=np.float32).ravel()
        else:
            if self._enc is None:
                from .encoders import SiglipEncoder
                self._enc = SiglipEncoder(self._encoder_name, device=self._device)
            v = self._enc.encode_text([query]).cpu().numpy()[0].astype(np.float32)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    def search(self, query: str | np.ndarray, k: int = 20, *,
               video: str | None = None, exact: bool = False,
               nprobes: int = 50, min_gap_s: float = 30.0,
               per_video: int | None = None) -> list[CollectionHit]:
        """The k best moments anywhere in the collection.

        video       restrict to one video -- a filter, not a separate index,
                    useful once a first search says where to look
        exact       scan everything instead of using the vector index. Slower,
                    and what to compare against when deciding whether the
                    approximation is costing you anything
        min_gap_s   collapse hits closer together than this within one video.
                    Without it a query returns the same moment five times: at
                    1 fps consecutive frames are near-identical, so the top of
                    any ranking is a run rather than five findings. 0 disables
        per_video   at most this many hits from any one video, so one
                    well-matching video cannot fill the whole result
        """
        q = self._query_vector(query)
        # over-fetch, because collapsing runs removes rows and the caller asked
        # for k distinct moments rather than k rows
        want = k if (min_gap_s <= 0 and per_video is None) else k * 12
        s = (self.table.search(q, vector_column_name="vector")
             .metric("cosine").limit(want))
        if video is not None:
            s = s.where(f"video = {_sql_str(video)}")
        if exact:
            s = s.bypass_vector_index()
        else:
            s = s.nprobes(nprobes)
        # lancedb returns cosine DISTANCE; similarity is what the rest of the
        # library reports, and mixing the two silently inverts a ranking
        hits = [CollectionHit(video=r["video"], time=float(r["ts"]),
                              score=1.0 - float(r["_distance"]))
                for r in s.to_list()]
        return self._collapse(hits, k, min_gap_s, per_video)

    @staticmethod
    def _collapse(hits: list[CollectionHit], k: int, min_gap_s: float,
                  per_video: int | None) -> list[CollectionHit]:
        """Keep the best hit of each run, in score order."""
        kept: list[CollectionHit] = []
        seen: dict[str, list[float]] = {}
        for h in hits:                       # already sorted best-first
            times = seen.setdefault(h.video, [])
            if per_video is not None and len(times) >= per_video:
                continue
            if min_gap_s > 0 and any(abs(h.time - t) < min_gap_s for t in times):
                continue
            times.append(h.time)
            kept.append(h)
            if len(kept) >= k:
                break
        return kept

    def recall_at(self, queries: list, k: int = 20, nprobes: int = 50) -> float:
        """What the approximate index costs, on your data rather than in general.

        Runs each query both ways and reports the share of the exact top-k that
        the approximate search also returned. Worth doing once per corpus:
        nprobes trades recall for latency and the right value is a property of
        how your vectors are distributed.
        """
        got = 0
        for query in queries:
            q = self._query_vector(query)
            a = {(h.video, h.time) for h in self.search(q, k, exact=True)}
            b = {(h.video, h.time) for h in self.search(q, k, nprobes=nprobes)}
            got += len(a & b) / max(1, len(a))
        return got / max(1, len(queries))
