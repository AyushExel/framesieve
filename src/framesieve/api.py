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

    the index is a sidecar     Indexing writes a Lance dataset next to the video.
                               It is small (about 11 MB per hour), portable, and
                               reading it needs no GPU and no model.

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
                and NOT comparable across queries -- nor across sources
    source      "visual" if a frame matched, "speech" if the transcript did,
                "both" when the two landed on the same moment
    text        the transcript line, when speech matched
    vlm_score   log-odds from the expensive model when `confirm=True`, else None.
                0 is a coin flip, +2 is about 7:1 for yes, -2 the reverse.
    """

    time: float
    score: float
    source: str = "visual"
    text: str | None = None
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
        t = "" if not self.text else f" {self.text[:40]!r}"
        return (f"<Hit {self.timecode} {self.source} "
                f"score={self.score:.3f}{v}{t}>")


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
                   fps: float = 1.0) -> str:
    """Where the sidecar for this (video, encoder, fps) lives.

    The encoder and rate are in the filename on purpose: an index built with a
    different encoder is not interchangeable, and silently reusing one would
    produce plausible nonsense.

    One path whether or not the index carries frames: `store=True` adds a blob
    column to the same dataset rather than writing a second file.
    """
    stem = os.path.splitext(video)[0]
    return f"{stem}.framesieve-{encoder}-{fps:g}fps.lance"


def _load_timed_text(video: str | None) -> dict:
    """Whatever timed-text passes have been run for this video.

    Each is a sibling dataset written by its own pass, so any subset of them can
    exist -- a video can have a transcript and no OCR, or the reverse.
    """
    from .audio import speech_path_for
    from .ocr import text_path_for
    from .timedtext import TimedTextIndex

    out: dict = {}
    if not video:
        return out
    for kind, path in (("speech", speech_path_for(video)),
                       ("text", text_path_for(video))):
        if os.path.exists(path):
            out[kind] = TimedTextIndex.load(path)
    return out


def _has_frames(path: str) -> bool:
    """Does this index carry the frames themselves, or only their embeddings?"""
    import lance

    try:
        return "jpeg" in lance.dataset(path).schema.names
    except Exception:
        return False


class VideoIndex:
    """A searchable video.

    Get one from `framesieve.open`, `framesieve.index` or `framesieve.load`.
    """

    def __init__(self, frame_index: FrameIndex, video: str | None = None,
                 encoder: str = DEFAULT_ENCODER, vlm: str = DEFAULT_VLM,
                 path: str | None = None, device: str | None = None,
                 store=None, timed_text=None):
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
        # {kind: TimedTextIndex} -- "speech" from Whisper, "text" from OCR.
        # Kept beside the frames rather than merged into them: each is produced
        # by its own pass and any of them can exist without the others.
        self._tt = dict(timed_text or {})
        self._text_enc = None
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

        Stored as float16, because that halves the sidecar and costs nothing
        measurable. Returned as float32 so arithmetic on them behaves -- and
        cached, because that cast was the entire cost of a search: 17.13 ms of
        17.42 ms on a 4.5-hour video, against 0.03 ms for the matmul it feeds.
        Paying it once per index instead of once per query is a 500x difference
        on everything after the first.

        The cache costs RAM: 4 bytes a dimension a frame, so about 50 MB for a
        4.5-hour video and 1.1 GB for a hundred hours. Past a few hundred hours
        that is the wrong trade and `Collection` is the answer -- see
        docs/scaling.md.
        """
        return self._index.emb

    @property
    def sources(self) -> list[str]:
        """What this index can be searched by: always "visual", plus "speech"
        and "text" when those passes were run."""
        return ["visual"] + [k for k, v in sorted(self._tt.items()) if len(v)]

    @property
    def has_speech(self) -> bool:
        return "speech" in self.sources

    @property
    def has_text(self) -> bool:
        """Was an OCR pass run over the frames?"""
        return "text" in self.sources

    @property
    def speech(self):
        """The transcript index, or None. Its `.segments` are the lines."""
        return self._tt.get("speech")

    @property
    def text(self):
        """The on-screen-text index, or None."""
        return self._tt.get("text")

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
               source: str | list[str] | None = None, question: str | None = None,
               strategy: str = "segment_adaptive", tokens_per_frame: int = 64,
               merge_gap_s: float = 10.0, seed: int = 0) -> SearchResults:
        """Find the k moments most likely to match `query`.

        query            what to look for, phrased as a caption ("a dark tunnel")
                         rather than a question -- the retrieval encoder is
                         caption-trained and the difference is worth real accuracy
        k                how many candidates to return, and with confirm=True how
                         many calls the expensive model makes
        confirm          show the surviving frames to a vision-language model and
                         return its verdict. About 30 ms a frame; without it a
                         "hit" only means "looks similar"
        source           which signals to search: "visual" (frames), "speech"
                         (the transcript), "text" (what is written on screen), or
                         a list of them. None searches everything the index has;
                         `.sources` says what that is
        question         the yes/no question put to the VLM. Defaults to
                         "Does this frame show: {query}?"
        strategy         how visual candidates are spread over the video
        merge_gap_s      hits from different sources this close together are the
                         same moment, and come back as one result whose `source`
                         names all of them
        """
        want = self._resolve_sources(source)

        text_hits = {kind: self._search_timed_text(kind, query, k)
                     for kind in want if kind != "visual"}
        if "visual" not in want:
            merged = self._merge(list(text_hits.values()), k, merge_gap_s)
            return SearchResults(query, merged, {}, k, "+".join(want), False)

        vis = self._search_visual(query, k, confirm=confirm, question=question,
                                  strategy=strategy,
                                  tokens_per_frame=tokens_per_frame, seed=seed)
        if not text_hits:
            return vis
        merged = self._merge([list(vis)] + list(text_hits.values()), k,
                             merge_gap_s)
        return SearchResults(query, merged, vis.timings, k, vis.strategy,
                             vis.confirmed)

    def _resolve_sources(self, source) -> list[str]:
        have = self.sources
        if source is None:
            return have
        want = [source] if isinstance(source, str) else list(source)
        for s in want:
            if s not in ("visual", "speech", "text"):
                raise ValueError(
                    f"source must be 'visual', 'speech', 'text' or a list of "
                    f"them, got {s!r}")
            if s not in have:
                how = {"speech": "audio=True", "text": "ocr=True"}[s]
                raise ValueError(
                    f"this index has no {s!r}. Rebuild with "
                    f"framesieve.index(video, {how}) or `framesieve index "
                    f"--{'audio' if s == 'speech' else 'ocr'}`.")
        return want

    def _search_visual(self, query, k, *, confirm, question, strategy,
                       tokens_per_frame, seed) -> SearchResults:
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
                    source="visual",
                    vlm_score=(None if res.vlm_score is None
                               else float(res.vlm_score[i])))
                for i in order]
        return SearchResults(query, hits, res.timings, k, strategy,
                             res.vlm_score is not None)

    def _search_timed_text(self, kind: str, query: str, k: int) -> list[Hit]:
        """Rank one timed-text source. Both of them use a sentence encoder,
        because SigLIP's text tower is built to sit beside images and is a poor
        text-to-text matcher."""
        from .timedtext import DEFAULT_TEXT_ENCODER, TextEncoder

        idx = self._tt[kind]
        if self._text_enc is None:
            self._text_enc = TextEncoder(
                idx.meta.get("text_encoder") or DEFAULT_TEXT_ENCODER,
                device=self._device)
        q = self._text_enc.encode([query], query=True)[0]
        sims = idx.emb @ q
        order = np.argsort(-sims)[:k]
        return [Hit(time=float(idx.segments[i].start), score=float(sims[i]),
                    source=kind, text=idx.segments[i].text) for i in order]

    @staticmethod
    def _merge(rankings: list[list[Hit]], k: int, gap_s: float) -> list[Hit]:
        """Combine several rankings without comparing their scores.

        A similarity against a frame, against a spoken sentence and against a
        line of on-screen text are three different quantities. Ordering them
        together by score would be meaningless. What IS comparable is rank --
        first place is first place -- and what is informative is agreement: when
        two signals point at the same second, that moment is a better candidate
        than either list's leader alone.

        So: group hits that fall within `gap_s`, name every source that found
        each group, and order by best rank with one place off per extra source
        that agreed.
        """
        flat = [(rank, h) for ranking in rankings
                for rank, h in enumerate(ranking)]
        flat.sort(key=lambda x: (x[0], x[1].time))

        # Group ACROSS sources only. Two visual hits a few seconds apart are a
        # separate question -- the selection strategy already decided they were
        # both worth returning -- and folding them together here would silently
        # drop one.
        groups: list[list[tuple[int, Hit]]] = []
        for rank, h in flat:
            for g in groups:
                if h.source in {o.source for _, o in g}:
                    continue
                if any(abs(h.time - other.time) <= gap_s for _, other in g):
                    g.append((rank, h))
                    break
            else:
                groups.append([(rank, h)])

        out: list[tuple[float, Hit]] = []
        for g in groups:
            best_rank = min(r for r, _ in g)
            lead = min(g, key=lambda x: x[0])[1]
            srcs = sorted({h.source for _, h in g})
            text = next((h.text for _, h in g if h.text), None)
            visual = next((h for _, h in g if h.source == "visual"), None)
            out.append((
                best_rank - (len(srcs) - 1),
                Hit(time=(visual or lead).time,
                    score=(visual or lead).score,
                    source="+".join(srcs) if len(srcs) > 1 else srcs[0],
                    text=text,
                    vlm_score=(visual.vlm_score if visual else None))))
        out.sort(key=lambda x: x[0])
        return [h for _, h in out[:k]]

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
          store: bool = False, audio: bool = False, ocr: bool = False,
          ocr_every: str = "segment", language: str | None = None,
          save: bool = True, batch: int = 256,
          size: int = 256, segment_tau: float = 0.0,
          pixel_gate_tau: float = 0.0, start: float = 0.0,
          duration: float = 0.0, gpu_decode: bool = False, seed: int = 0,
          jpeg_quality: int = 90, verbose: bool = False) -> VideoIndex:
    """Index a video: decode at `fps`, embed every frame, write a sidecar.

    Costs roughly 15 seconds and 11 MB per hour of video on one GPU. Runs once;
    every search after this reads the sidecar.

    `audio=True` also transcribes the video with Whisper and indexes the timed
    segments, so `search(..., source="speech")` can reach things that were said
    rather than shown. Runs at about 11x realtime and writes a sibling dataset;
    skipped with a warning when the file has no audio track.

    `ocr=True` reads the text on screen and indexes it, so `source="text"` can
    reach a caption or a slide title. The most expensive pass here at roughly
    7 minutes per hour of video, which `ocr_every="segment"` (the default) cuts
    by reading one frame per shot instead of every frame.

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

    tt: dict = {}
    if audio:
        sp = _index_audio(video, device, language, verbose)
        if sp is not None:
            tt["speech"] = sp

    if store:
        from .store import FrameStore, build_store
        out = index_path_for(video, encoder, fps)
        build_store(video, enc, out, target_fps=fps, size=size, batch=batch,
                    segment_tau=segment_tau, jpeg_quality=jpeg_quality,
                    gpu_decode=gpu_decode, seed=seed)
        fs_ = FrameStore(out)
        return VideoIndex(fs_.to_frame_index(), video=video, encoder=encoder,
                          vlm=vlm, path=out, device=device, store=fs_,
                          timed_text=tt)

    fi = build_index(video, enc, target_fps=fps, batch=batch,
                     size=size, pixel_gate_tau=pixel_gate_tau,
                     segment_tau=segment_tau, start_s=start,
                     duration_s=duration, gpu_decode=gpu_decode, seed=seed,
                     verbose=verbose)
    vi = VideoIndex(fi, video=video, encoder=encoder, vlm=vlm, device=device,
                    timed_text=tt)
    if save:
        vi.save(index_path_for(video, encoder, fps))
    if ocr:
        # after the frame index exists, because the OCR pass uses its
        # segmentation to decide which frames are worth reading
        from .ocr import build_text_index, text_path_for

        ti = build_text_index(video, index=fi, fps=fps, every=ocr_every,
                              device=device, verbose=verbose)
        ti.save(text_path_for(video))
        vi._tt["text"] = ti
    return vi


def _index_audio(video: str, device, language, verbose):
    """Transcribe and embed, or say clearly why not.

    Silent footage is common and Whisper does not merely return nothing for it,
    it hallucinates -- so a missing audio track is checked for rather than
    discovered.
    """
    from .audio import build_speech_index, has_audio, speech_path_for

    if not has_audio(video):
        print(f"warning: {os.path.basename(video)} has no audio track; "
              f"indexing frames only")
        return None
    sp = build_speech_index(video, device=device, language=language,
                            verbose=verbose)
    sp.save(speech_path_for(video))
    return sp


def load(path_or_video: str, *, video: str | None = None,
         encoder: str = DEFAULT_ENCODER, fps: float = 1.0,
         vlm: str = DEFAULT_VLM, device: str | None = None) -> VideoIndex:
    """Load an existing index, given either the sidecar or the video it came from.

    Needs no GPU and no model: an index is a small array file, and reading it is
    the cheap half of everything this library does.
    """
    p = (path_or_video if path_or_video.endswith(".lance")
         else index_path_for(path_or_video, encoder, fps))
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"no index at {p}. Build one with framesieve.index({path_or_video!r}) "
            f"or `framesieve index {path_or_video}`.")

    # Both forms are Lance datasets; only one carries the frames. Deciding on
    # the schema rather than on whether a constructor raises, because FrameStore
    # opens a frameless dataset perfectly happily and then fails much later, at
    # the point someone asks for a frame.
    st = None
    if _has_frames(p):
        from .store import FrameStore
        st = FrameStore(p)
        fi = st.to_frame_index()
    else:
        fi = FrameIndex.load(p)
    src = video or (path_or_video if not path_or_video.endswith(".lance")
                    else fi.stats.video)
    return VideoIndex(fi, video=src, encoder=encoder, vlm=vlm, path=p,
                      device=device, store=st,
                      timed_text=_load_timed_text(src))


def open(video: str, *, encoder: str = DEFAULT_ENCODER, fps: float = 1.0,
         vlm: str = DEFAULT_VLM, device: str | None = None,
         rebuild: bool = False, **kwargs) -> VideoIndex:
    """Load this video's index, building it first if it does not exist.

    The one call most programs want. Shadows the builtin inside this module
    only; as `framesieve.open(...)` there is no ambiguity.
    """
    p = index_path_for(video, encoder, fps)
    if os.path.exists(p) and not rebuild:
        return load(p, video=video, encoder=encoder, fps=fps, vlm=vlm,
                    device=device)
    return index(video, encoder=encoder, fps=fps, vlm=vlm, device=device,
                 **kwargs)
