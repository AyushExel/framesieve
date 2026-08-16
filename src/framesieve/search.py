"""The cascade: pick which frames deserve the expensive model, then ask it.

Selection strategies are deliberately kept as siblings behind one interface, so
the ablation is a change of one string rather than a change of pipeline. They
differ only in how they spend a fixed budget of VLM calls:

  uniform   spend it evenly over wall-clock time, ignoring the index entirely.
            This is what everybody does, and it is stronger than people expect.
  topk      spend it on the highest-scoring frames. Simple, and on real video
            usually wrong: the top 100 frames are often 100 near-copies of one
            moment.
  nms       top-k with temporal non-maximum suppression -- take the best frame,
            then forbid anything within +/- w seconds of it, repeat.
  segment   top-k over the index's redundancy-collapsed segments, one frame per
            segment. The collapse is precomputed, so this costs nothing at query
            time -- but its granularity is frozen at index time, and the ablation
            shows the right granularity depends on the budget.
  segment_adaptive
            the same, except the video is cut into a number of segments
            proportional to the budget, at query time. Cutting is O(N log N) over
            cached adjacent similarities, so "free" is still true.

`budget` is always the number of VLM calls, because that is the only cost that
matters once the index exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:                      # pragma: no cover
    from .encoders import SiglipEncoder
from .index import FrameIndex

STRATEGIES = ("uniform", "topk", "nms", "segment", "segment_adaptive")

# How many segments to cut the video into, as a multiple of the VLM budget.
# The ablation on the held-out cab-ride video showed the best granularity is not
# a constant: at 32 calls a coarse segmentation wins, at 128 a finer one does.
# What is roughly constant is the *ratio* of segments to calls, which is what
# this expresses. It interpolates between two familiar behaviours -- at 1 you
# take one frame from every segment, which is uniform sampling in content space;
# at very large values the segments stop constraining anything and you are back
# to plain top-k.
#
# Chosen as 8 from the sweep in scripts/ablate.py on the cab-ride video, and
# then held fixed for Video-MME so the benchmark stays a checkpoint rather than
# something this was tuned against. The optimum is broad -- anything from 4 to 16
# is within noise of it -- which is the useful part: this is not a knob that
# needs care.
SEGMENT_FACTOR = 8.0


@dataclass
class Candidates:
    ts: np.ndarray
    cheap_score: np.ndarray
    strategy: str
    n_considered: int


@dataclass
class SearchResult:
    query: str
    question: str
    strategy: str
    budget: int
    ts: np.ndarray
    cheap_score: np.ndarray
    vlm_score: np.ndarray | None = None
    timings: dict = field(default_factory=dict)

    def ranked(self, by: str = "vlm") -> np.ndarray:
        s = self.vlm_score if (by == "vlm" and self.vlm_score is not None) else self.cheap_score
        return np.argsort(-s)


def select_candidates(index: FrameIndex, query_emb: np.ndarray, budget: int,
                      strategy: str = "segment", nms_window_s: float | None = None,
                      seed: int = 0, segment_factor: float = SEGMENT_FACTOR
                      ) -> Candidates:
    """Choose `budget` timestamps to spend VLM calls on.

    nms_window_s=None makes the suppression window adapt to the budget:
    span / (2 * budget). A fixed window silently caps how much budget NMS can
    spend -- a 30 s window on a 10 minute clip cannot produce more than ~20
    candidates, so every budget above that returns the same answer and the recall
    curve goes flat for reasons that have nothing to do with retrieval quality.
    """
    ts = index.ts
    n = len(ts)
    if budget >= n:
        budget = n

    if strategy == "uniform":
        # random phase so that repeated seeds give an honest spread; a fixed
        # phase would make uniform sampling look artificially stable
        rng = np.random.default_rng(seed)
        edges = np.linspace(0, n, budget + 1)
        pick = np.floor(edges[:-1] + rng.uniform(0, 1, budget) * np.diff(edges)).astype(int)
        pick = np.clip(np.unique(pick), 0, n - 1)
        sim = index.emb.astype(np.float32) @ query_emb
        return Candidates(ts=ts[pick], cheap_score=sim[pick], strategy=strategy,
                          n_considered=n)

    sim = index.emb.astype(np.float32) @ query_emb

    if strategy == "topk":
        pick = np.argsort(-sim)[:budget]
        return Candidates(ts=ts[pick], cheap_score=sim[pick], strategy=strategy,
                          n_considered=n)

    if strategy == "nms":
        span = float(ts[-1] - ts[0]) if n > 1 else 1.0
        w = nms_window_s if nms_window_s is not None else span / (2.0 * max(1, budget))
        order = np.argsort(-sim)
        taken: list[int] = []
        blocked = np.zeros(n, dtype=bool)
        for i in order:
            if blocked[i]:
                continue
            taken.append(int(i))
            if len(taken) >= budget:
                break
            lo = np.searchsorted(ts, ts[i] - w, "left")
            hi = np.searchsorted(ts, ts[i] + w, "right")
            blocked[lo:hi] = True
        pick = np.array(taken, dtype=int)
        return Candidates(ts=ts[pick], cheap_score=sim[pick], strategy=strategy,
                          n_considered=n)

    if strategy in ("segment", "segment_adaptive"):
        if strategy == "segment_adaptive":
            # granularity is chosen per query, from the budget, instead of being
            # frozen at index time. Cutting is O(N log N) over cached adjacent
            # similarities, so this is free at query time.
            seg_id = index.cut_into(int(round(segment_factor * budget)))
            bounds = np.flatnonzero(np.diff(seg_id)) + 1
            starts = np.concatenate([[0], bounds])
            ends = np.concatenate([bounds, [len(seg_id)]])
        else:
            starts, ends, _, _ = index.segments()
        # Rank segments by their best frame, then take frames round-robin: every
        # segment's best before any segment's second-best. When the budget
        # exceeds the segment count this degrades gracefully towards top-k
        # *within* the ranked segments, instead of falling back to a global top-k
        # that would re-introduce exactly the redundancy segmentation removed.
        seg_order_by_score: list[np.ndarray] = []
        best_sc = np.empty(len(starts), dtype=np.float32)
        for j, (s, e) in enumerate(zip(starts, ends)):
            local = np.argsort(-sim[s:e]) + s
            seg_order_by_score.append(local)
            best_sc[j] = sim[local[0]]
        seg_rank = np.argsort(-best_sc)

        pick_list: list[int] = []
        depth = 0
        while len(pick_list) < budget:
            added = False
            for j in seg_rank:
                if len(pick_list) >= budget:
                    break
                cand_list = seg_order_by_score[j]
                if depth < len(cand_list):
                    pick_list.append(int(cand_list[depth]))
                    added = True
            if not added:
                break
            depth += 1
        pick = np.array(pick_list, dtype=int)
        return Candidates(ts=ts[pick], cheap_score=sim[pick], strategy=strategy,
                          n_considered=len(starts))

    raise ValueError(f"unknown strategy {strategy!r}; have {STRATEGIES}")


class CascadeSearcher:
    def __init__(self, index: FrameIndex, encoder: SiglipEncoder,
                 vlm=None, fetcher=None):
        self.index = index
        self.encoder = encoder
        self.vlm = vlm
        self.fetcher = fetcher
        self._text_cache: dict[str, np.ndarray] = {}

    def text_embedding(self, query: str) -> np.ndarray:
        if query not in self._text_cache:
            self._text_cache[query] = (
                self.encoder.encode_text([query]).cpu().numpy()[0].astype(np.float32))
        return self._text_cache[query]

    def search(self, query: str, budget: int, question: str | None = None,
               strategy: str = "segment", nms_window_s: float = 30.0,
               seed: int = 0, refine: bool = True) -> SearchResult:
        import time
        t0 = time.perf_counter()
        qe = self.text_embedding(query)
        cand = select_candidates(self.index, qe, budget, strategy=strategy,
                                 nms_window_s=nms_window_s, seed=seed)
        t_sel = time.perf_counter() - t0

        res = SearchResult(query=query, question=question or query,
                           strategy=strategy, budget=budget, ts=cand.ts,
                           cheap_score=cand.cheap_score,
                           timings={"select_s": t_sel})
        if not refine or self.vlm is None or self.fetcher is None:
            return res

        t1 = time.perf_counter()
        got_ts, frames = self.fetcher.fetch(cand.ts.tolist())
        t_fetch = time.perf_counter() - t1

        t2 = time.perf_counter()
        scores = []
        B = 16
        for i in range(0, len(frames), B):
            scores.append(self.vlm.score(list(frames[i:i + B]), res.question))
        t_vlm = time.perf_counter() - t2

        # the fetcher may drop undecodable frames, so realign the cheap scores by
        # timestamp rather than by position -- candidates are ordered by score,
        # not by time, so interpolating here would silently scramble them
        lut = {float(t): float(s) for t, s in zip(cand.ts, cand.cheap_score)}
        res.ts = got_ts
        res.cheap_score = np.array([lut[float(t)] for t in got_ts], dtype=np.float32)
        res.vlm_score = np.concatenate(scores) if scores else np.zeros(0, np.float32)
        res.timings.update({"fetch_s": t_fetch, "vlm_s": t_vlm,
                            "total_s": time.perf_counter() - t0})
        return res
