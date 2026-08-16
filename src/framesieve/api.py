"""The public API: index a video once, then search it.

    import framesieve as fs

    video = fs.open("holiday.mp4")            # index if needed, else load
    for hit in video.search("a dark tunnel"):
        print(hit.timecode, hit.score)

Everything below is a thin, stable layer over `framesieve.index`,
`framesieve.search`, `framesieve.encoders` and `framesieve.vlm`. Those stay
importable and are what the research scripts use; this module is what should not
change under you.

Three ideas are worth knowing before reading further.

    the index is a sidecar     Indexing writes a file next to the video. It is
                               small (about 5 MB per hour), portable, and reading
                               it needs no GPU and no model.

    models load lazily         Opening an index costs no GPU. The image encoder
                               loads on the first search, the vision-language
                               model only if you ask for `confirm=True`.

    a GPU is optional          Everything picks CUDA when there is one, Apple
                               silicon when there is one, and CPU otherwise.
                               Pass device= to override.

    confirm= is the expensive  Retrieval alone is a matrix multiply, about a
    half                       millisecond. `confirm=True` fetches the surviving
                               frames and shows them to a real VLM, which costs
                               roughly 30 ms each and is what makes the answer
                               trustworthy rather than merely similar.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from .index import FrameIndex, IndexStats, build_index
from .search import STRATEGIES, CascadeSearcher

__all__ = ["Hit", "SearchResults", "VideoIndex", "index", "load", "open",
           "index_path_for", "DEFAULT_ENCODER", "DEFAULT_VLM"]

DEFAULT_ENCODER = "siglip2-base-224"
DEFAULT_VLM = "qwen2.5-vl-7b"


def timecode(seconds: float) -> str:
    """Seconds as h:mm:ss, which is what a video player wants."""
    s = max(0.0, float(seconds))
    h, rem = divmod(int(s), 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}"


@dataclass(frozen=True)
class Hit:
    """One moment in the video.

    time        seconds from the start
    score       retrieval similarity, roughly -1..1, comparable within a query
                and NOT comparable across queries
    vlm_score   log-odds from the expensive model when `confirm=True`, else None.
                0 is a coin flip, +2 is about 7:1 for yes, -2 the reverse.
    """

    time: float
    score: float
    vlm_score: float | None = None

    @property
    def timecode(self) -> str:
        return timecode(self.time)

    @property
    def confirmed(self) -> bool | None:
        """True/False once a VLM has looked, None if it has not."""
        return None if self.vlm_score is None else self.vlm_score > 0.0

    def __repr__(self) -> str:
        v = "" if self.vlm_score is None else f" vlm={self.vlm_score:+.2f}"
        return f"<Hit {self.timecode} score={self.score:.3f}{v}>"


class SearchResults(Sequence[Hit]):
    """An ordered, sliceable list of `Hit` that also carries the timings.

    Ordered by the expensive model's verdict when there is one, by retrieval
    similarity otherwise -- so `results[0]` is always the best answer available.
    """

    def __init__(self, query: str, hits: list[Hit], timings: dict,
                 budget: int, strategy: str, confirmed: bool):
        self.query = query
        self._hits = hits
        self.timings = timings
        self.budget = budget
        self.strategy = strategy
        self.confirmed = confirmed

    def __len__(self) -> int:
        return len(self._hits)

    def __getitem__(self, i):
        return self._hits[i] if isinstance(i, int) else SearchResults(
            self.query, self._hits[i], self.timings, self.budget,
            self.strategy, self.confirmed)

    def __iter__(self) -> Iterator[Hit]:
        return iter(self._hits)

    def above(self, threshold: float = 0.0) -> SearchResults:
        """Only the hits the VLM scored above `threshold`.

        Meaningless without `confirm=True`, and says so rather than silently
        filtering on a similarity that has no absolute scale.
        """
        if not self.confirmed:
            raise ValueError(
                "above() needs VLM scores; call search(..., confirm=True). "
                "Retrieval similarity has no absolute scale, so a threshold on "
                "it would mean nothing.")
        return SearchResults(self.query,
                             [h for h in self._hits if (h.vlm_score or 0.0) > threshold],
                             self.timings, self.budget, self.strategy, True)

    @property
    def times(self) -> np.ndarray:
        return np.array([h.time for h in self._hits], dtype=np.float64)

    def to_dicts(self) -> list[dict]:
        return [{"time": h.time, "timecode": h.timecode, "score": h.score,
                 "vlm_score": h.vlm_score} for h in self._hits]

    @property
    def latency_ms(self) -> float:
        return 1000.0 * sum(self.timings.get(k, 0.0)
                            for k in ("select_s", "fetch_s", "vlm_s"))

    def __repr__(self) -> str:
        return (f"<SearchResults {self.query!r} {len(self._hits)} hits, "
                f"{self.latency_ms:.0f} ms"
                f"{', VLM-confirmed' if self.confirmed else ''}>")


def index_path_for(video: str, encoder: str = DEFAULT_ENCODER,
                   fps: float = 1.0, store: bool = False) -> str:
    """Where the sidecar for this (video, encoder, fps) lives.

    The encoder and rate are in the filename on purpose: an index built with a
    different encoder is not interchangeable, and silently reusing one would
    produce plausible nonsense.

    `store=True` names the frame-store form, which keeps the frames themselves
    beside the embeddings and so ends in .lance rather than .npz.
    """
    stem = os.path.splitext(video)[0]
    ext = "lance" if store else "npz"
    return f"{stem}.framesieve-{encoder}-{fps:g}fps.{ext}"


class VideoIndex:
    """A searchable video.

    Get one from `framesieve.open`, `framesieve.index` or `framesieve.load`.
    """

    def __init__(self, frame_index: FrameIndex, video: str | None = None,
                 encoder: str = DEFAULT_ENCODER, vlm: str = DEFAULT_VLM,
                 path: str | None = None, device: str | None = None,
                 store=None):
        self._index = frame_index
        self.video = video or frame_index.stats.video
        self.path = path
        self._encoder_name = encoder
        self._vlm_name = vlm
        self._device = device
        # a FrameStore, when the index was built with store=True. It holds the
        # frames as well as the embeddings, so it can serve them back by
        # byte-range read instead of seeking the video: 0.9 ms a frame against
        # 14.5 ms, measured. It also means confirm= and frames() work without
        # the source video present at all.
        self._store = store
        self._searcher: CascadeSearcher | None = None
        self._vlm = None
        self._fetcher = None

    # -- properties --------------------------------------------------------

    @property
    def stats(self) -> IndexStats:
        """How the index was built: encoder, revision, frame count, throughput."""
        return self._index.stats

    @property
    def duration(self) -> float:
        return float(self._index.stats.duration_s)

    @property
    def times(self) -> np.ndarray:
        """The timestamp of every indexed frame, in seconds."""
        return self._index.ts.astype(np.float64)

    @property
    def embeddings(self) -> np.ndarray:
        """The frame embeddings, L2-normalised, one row per indexed frame.

        Returned as float32 so arithmetic on them behaves; they are stored as
        float16 because that halves the sidecar and costs nothing measurable.
        """
        return self._index.emb.astype(np.float32)

    @property
    def frame_index(self) -> FrameIndex:
        """The underlying object, for the lower-level modules."""
        return self._index

    def __len__(self) -> int:
        return len(self._index.ts)

    def __repr__(self) -> str:
        return (f"<VideoIndex {os.path.basename(str(self.video))} "
                f"{len(self):,} frames, {self.duration/3600:.2f} h, "
                f"{self._encoder_name}>")

    # -- lazy model loading ------------------------------------------------

    def _get_searcher(self) -> CascadeSearcher:
        if self._searcher is None:
            try:
                from .encoders import SiglipEncoder
            except ImportError as exc:  # pragma: no cover - environment
                raise ImportError(
                    "searching by text needs the encoder, which needs torch. "
                    "Reading an index does not.\n"
                    "  Either install torch, or encode the query elsewhere and "
                    "pass the vector:  video.score(query_vector)"
                ) from exc
            self._searcher = CascadeSearcher(
                self._index, SiglipEncoder(self._encoder_name,
                                           device=self._device))
        return self._searcher

    def _query_vector(self, query: str | np.ndarray) -> np.ndarray:
        """A text query becomes a vector; a vector is used as given.

        Accepting a precomputed vector is what makes the torch-free path useful
        rather than a curiosity: encode text on a GPU box, ship the vector, and
        rank anywhere numpy runs.
        """
        if isinstance(query, str):
            return self._get_searcher().text_embedding(query)
        v = np.asarray(query, dtype=np.float32).ravel()
        if v.shape[0] != self._index.emb.shape[1]:
            raise ValueError(
                f"query vector has {v.shape[0]} dimensions but this index has "
                f"{self._index.emb.shape[1]}; it was built with "
                f"{self._index.stats.encoder!r}")
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    def _ensure_refine(self, tokens_per_frame: int) -> None:
        """Load the expensive model and the frame fetcher, once."""
        s = self._get_searcher()
        if s.vlm is None:
            from .vlm import QwenYesNoScorer
            px = tokens_per_frame * 28 * 28 * 4
            s.vlm = QwenYesNoScorer(self._vlm_name, device=self._device,
                                    max_pixels=px,
                                    min_pixels=min(px, 64 * 28 * 28))
        if s.fetcher is None:
            if self._store is not None:
                s.fetcher = self._store
            elif self.video and os.path.exists(str(self.video)):
                from .fetch import FrameFetcher
                s.fetcher = FrameFetcher(str(self.video), workers=16)
            else:
                raise FileNotFoundError(
                    f"confirm=True needs the frames. This index has no frame "
                    f"store, and the source video {self.video!r} is not there. "
                    f"Pass video= when loading an index whose source has moved, "
                    f"or build the index with store=True.")

    # -- the actual API ----------------------------------------------------

    def score(self, query: str | np.ndarray) -> np.ndarray:
        """Retrieval similarity for every indexed frame, aligned with `times`.

        The raw material: use it when you want the whole curve rather than the
        top few, e.g. to plot where in the video a concept appears.

        `query` may be a precomputed, same-dimension embedding instead of text,
        in which case this needs numpy and nothing else.
        """
        return self.embeddings @ self._query_vector(query)

    def search(self, query: str, k: int = 32, *, confirm: bool = False,
               question: str | None = None, strategy: str = "segment_adaptive",
               tokens_per_frame: int = 64, seed: int = 0) -> SearchResults:
        """Find the k moments most likely to match `query`.

        query            what to look for, phrased as a caption ("a dark tunnel")
                         rather than a question -- the retrieval encoder is
                         caption-trained and the difference is worth real accuracy
        k                how many candidate frames to return, and with
                         confirm=True how many calls the expensive model makes
        confirm          show the surviving frames to a vision-language model and
                         return its verdict. Costs about 30 ms per frame; without
                         it a "hit" only means "looks similar"
        question         the yes/no question put to the VLM. Defaults to
                         "Does this frame show: {query}?"
        strategy         how candidates are spread over the video; see
                         framesieve.search.STRATEGIES. The default avoids
                         returning k near-copies of one moment
        """
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r}; "
                             f"have {list(STRATEGIES)}")
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        if confirm:
            self._ensure_refine(tokens_per_frame)

        res = self._get_searcher().search(
            query, budget=k, question=question or f"Does this frame show: {query}?",
            strategy=strategy, seed=seed, refine=confirm)

        order = res.ranked("vlm" if res.vlm_score is not None else "cheap")
        hits = [Hit(time=float(res.ts[i]), score=float(res.cheap_score[i]),
                    vlm_score=(None if res.vlm_score is None
                               else float(res.vlm_score[i])))
                for i in order]
        return SearchResults(query, hits, res.timings, k, strategy,
                             res.vlm_score is not None)

    def frames(self, times: Iterable[float] | SearchResults,
               size: int | None = None) -> list[np.ndarray]:
        """Fetch the actual pixels at these timestamps, as uint8 HWC arrays.

        Accepts a SearchResults directly, so `video.frames(hits[:4])` works.
        """
        if isinstance(times, SearchResults):
            times = times.times
        ts = [float(t) for t in times]
        if not ts:
            return []
        if self._store is not None:
            _, frames = self._store.fetch(ts)
            return list(frames)
        if not self.video or not os.path.exists(str(self.video)):
            raise FileNotFoundError(
                f"fetching frames needs either a frame store or the video "
                f"file, and {self.video!r} is not there")
        from .fetch import FrameFetcher
        f = FrameFetcher(str(self.video), size=size, workers=16)
        _, frames = f.fetch(ts)
        return list(frames)

    def save(self, path: str | None = None) -> str:
        p = path or self.path or index_path_for(str(self.video), self._encoder_name,
                                                self._index.stats.target_fps)
        self._index.save(p)
        self.path = p
        return p


# --------------------------------------------------------------------------
# module-level entry points
# --------------------------------------------------------------------------


def index(video: str, *, encoder: str = DEFAULT_ENCODER, fps: float = 1.0,
          vlm: str = DEFAULT_VLM, device: str | None = None,
          store: bool = False, save: bool = True, batch: int = 256,
          size: int = 256, segment_tau: float = 0.0,
          pixel_gate_tau: float = 0.0, start: float = 0.0,
          duration: float = 0.0, gpu_decode: bool = False, seed: int = 0,
          jpeg_quality: int = 90, verbose: bool = False) -> VideoIndex:
    """Index a video: decode at `fps`, embed every frame, write a sidecar.

    Costs roughly 15 seconds and 5 MB per hour of video on one GPU. Runs once;
    every search after this reads the sidecar.

    `store=True` also keeps every sampled frame as a JPEG beside its embedding.
    That makes confirm= about 15x faster at fetching frames (0.9 ms each against
    14.5 ms of ffmpeg seeking) and lets the index work without the source video
    present -- at roughly 55x the disk, 275 MB per hour against 5 MB, and about
    half the indexing throughput. Off by default because most of that disk buys
    nothing unless you use confirm= heavily. Needs `pip install pylance`.
    """
    if not os.path.exists(video):
        raise FileNotFoundError(video)
    from .encoders import SiglipEncoder
    enc = SiglipEncoder(encoder, device=device)

    if store:
        from .store import FrameStore, build_store
        out = index_path_for(video, encoder, fps, store=True)
        build_store(video, enc, out, target_fps=fps, size=size, batch=batch,
                    segment_tau=segment_tau, jpeg_quality=jpeg_quality,
                    gpu_decode=gpu_decode, seed=seed)
        fs_ = FrameStore(out)
        return VideoIndex(fs_.to_frame_index(), video=video, encoder=encoder,
                          vlm=vlm, path=out, device=device, store=fs_)

    fi = build_index(video, enc, target_fps=fps, batch=batch,
                     size=size, pixel_gate_tau=pixel_gate_tau,
                     segment_tau=segment_tau, start_s=start,
                     duration_s=duration, gpu_decode=gpu_decode, seed=seed,
                     verbose=verbose)
    vi = VideoIndex(fi, video=video, encoder=encoder, vlm=vlm, device=device)
    if save:
        vi.save(index_path_for(video, encoder, fps))
    return vi


def load(path_or_video: str, *, video: str | None = None,
         encoder: str = DEFAULT_ENCODER, fps: float = 1.0,
         vlm: str = DEFAULT_VLM, device: str | None = None) -> VideoIndex:
    """Load an existing index, given either the sidecar or the video it came from.

    Needs no GPU and no model: an index is a small array file, and reading it is
    the cheap half of everything this library does.
    """
    if path_or_video.endswith((".npz", ".lance")):
        p = path_or_video
    else:
        # a frame store carries everything the plain index does and more, so
        # prefer it when both are present
        lance_p = index_path_for(path_or_video, encoder, fps, store=True)
        npz_p = index_path_for(path_or_video, encoder, fps)
        p = lance_p if os.path.exists(lance_p) else npz_p
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"no index at {p}. Build one with framesieve.index({path_or_video!r}) "
            f"or `framesieve index {path_or_video}`.")

    if p.endswith(".lance"):
        from .store import FrameStore
        st = FrameStore(p)
        fi = st.to_frame_index()
    else:
        st, fi = None, FrameIndex.load(p)
    src = video or (path_or_video if not path_or_video.endswith((".npz", ".lance"))
                    else fi.stats.video)
    return VideoIndex(fi, video=src, encoder=encoder, vlm=vlm, path=p,
                      device=device, store=st)


def open(video: str, *, encoder: str = DEFAULT_ENCODER, fps: float = 1.0,
         vlm: str = DEFAULT_VLM, device: str | None = None,
         rebuild: bool = False, **kwargs) -> VideoIndex:
    """Load this video's index, building it first if it does not exist.

    The one call most programs want. Shadows the builtin inside this module
    only; as `framesieve.open(...)` there is no ambiguity.
    """
    existing = [q for q in (index_path_for(video, encoder, fps, store=True),
                            index_path_for(video, encoder, fps))
                if os.path.exists(q)]
    if existing and not rebuild:
        return load(existing[0], video=video, encoder=encoder, fps=fps,
                    vlm=vlm, device=device)
    return index(video, encoder=encoder, fps=fps, vlm=vlm, device=device,
                 **kwargs)
