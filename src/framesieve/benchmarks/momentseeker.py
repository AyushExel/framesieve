"""MomentSeeker under the protocol its paper and evaluation code actually use.

This is long-video moment retrieval: given a text query and a video averaging
over 500 s, return the time interval that answers it. Unlike the VideoQA
benchmarks, it is the same task framesieve is built for.

Protocol, taken from the paper (arXiv 2502.12558, §4.1) and cross-checked against
the released evaluation code:

  candidates   "We divide each video into fixed 10-second chunks without
               tailoring the segmentation to any specific model."
  R@1          the top-1 predicted chunk counts if its temporal IoU with any
               ground-truth moment exceeds 0.3.
  mAP@5        the released code computes a *graded* AP over the top 5, using
               IoU against the single interval spanning all ground truths as the
               relevance score, with no threshold:

                   cum = 0; s = 0
                   for pos, clip in enumerate(top5, 1):
                       iou  = IoU(clip, union_of_gt)
                       cum += iou
                       s   += (cum / pos) * iou
                   AP = s

               The paper describes a threshold-and-match formulation instead.
               They disagree, and the published numbers came from the code, so
               the code's version is what `map_at_5` implements. The paper's
               version is also provided as `map_at_5_matched` so the difference
               can be seen rather than argued about.

Baselines to beat, from the paper's Table 2 (Overall, retrieval-based):

    method            size    R@1    mAP@5
    UniIR             428M   11.2     16.9
    MM-Ret            148M   12.4     17.7
    CoVR              588M   13.0     18.5
    E5V               8.4B   14.3     20.1
    LanguageBind      428M   18.2     25.4
    InternVideo2        1B   19.7     26.6

Note those are the Overall column across text-, image- and video-conditioned
queries; this module evaluates the text-only (t2v) split, so the comparison is
indicative rather than exact.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

CHUNK_S = 10.0
IOU_THRESHOLD = 0.3

# paper Table 2, Overall / retrieval-based
PAPER_BASELINES = {
    "UniIR": (11.2, 16.9), "MM-Ret": (12.4, 17.7), "CoVR": (13.0, 18.5),
    "E5V": (14.3, 20.1), "LanguageBind": (18.2, 25.4), "InternVideo2": (19.7, 26.6),
}

# the paper groups the nine annotated tasks into three meta-tasks
META_TASK = {
    "Description Location": "global",
    "Action Recognition": "event", "Causal Reasoning": "event",
    "Anomaly Detection": "event",
    "Object Recognition": "object", "Attribute Recognition": "object",
    "OCR": "object", "Object Location": "object", "Spatial Relation": "object",
}


@dataclass
class MsQuery:
    text: str
    video_id: str
    video_path: str | None
    gt_intervals: list[tuple[float, float]]
    task: str

    @property
    def meta_task(self) -> str:
        return META_TASK.get(self.task, "other")


def load_queries(json_path: str, video_dir: str) -> list[MsQuery]:
    raw = json.load(open(json_path))
    out, missing = [], 0
    for r in raw:
        vid = os.path.basename(r["src_video_path"])
        p = os.path.join(video_dir, vid)
        if not os.path.exists(p):
            missing += 1
            p = None
        out.append(MsQuery(text=r["qry_text"], video_id=os.path.splitext(vid)[0],
                           video_path=p,
                           gt_intervals=[(float(a), float(b))
                                         for a, b in r["answering_time_interval"]],
                           task=r["task"]))
    if missing:
        print(f"  warning: {missing}/{len(out)} queries have no local video")
    return out


def chunks_for(duration_s: float, chunk_s: float = CHUNK_S) -> np.ndarray:
    """Fixed non-overlapping chunks, as the benchmark specifies."""
    n = max(1, int(np.ceil(duration_s / chunk_s)))
    starts = np.arange(n, dtype=np.float64) * chunk_s
    ends = np.minimum(starts + chunk_s, duration_s)
    return np.stack([starts, ends], axis=1)


def iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def gt_chunk_mask(chunks: np.ndarray, gts: Sequence[tuple[float, float]],
                  thr: float = IOU_THRESHOLD) -> np.ndarray:
    """Which chunks count as ground truth: IoU above threshold with any GT moment."""
    m = np.zeros(len(chunks), dtype=bool)
    for i, c in enumerate(chunks):
        m[i] = any(iou((c[0], c[1]), g) >= thr for g in gts)
    return m


def recall_at_k(ranked: np.ndarray, is_gt: np.ndarray, k: int = 1) -> float:
    return float(is_gt[ranked[:k]].any()) if len(ranked) else 0.0


def map_at_5(ranked: np.ndarray, chunks: np.ndarray,
             gts: Sequence[tuple[float, float]]) -> float:
    """The released evaluation code's graded AP. See module docstring."""
    if not len(gts) or not len(ranked):
        return 0.0
    union = (min(g[0] for g in gts), max(g[1] for g in gts))
    cum = 0.0
    s = 0.0
    for pos, ci in enumerate(ranked[:5], start=1):
        v = iou((chunks[ci][0], chunks[ci][1]), union)
        cum += v
        s += (cum / pos) * v
    return s


def map_at_5_matched(ranked: np.ndarray, chunks: np.ndarray,
                     gts: Sequence[tuple[float, float]],
                     thr: float = IOU_THRESHOLD) -> float:
    """The paper's description: threshold, greedy one-to-one GT matching."""
    if not len(gts) or not len(ranked):
        return 0.0
    unmatched = list(range(len(gts)))
    hits, precisions = 0, []
    for pos, ci in enumerate(ranked[:5], start=1):
        best, best_iou = None, thr
        for gi in unmatched:
            v = iou((chunks[ci][0], chunks[ci][1]), gts[gi])
            if v >= best_iou:
                best, best_iou = gi, v
        if best is not None:
            unmatched.remove(best)
            hits += 1
            precisions.append(hits / pos)
    return float(np.mean(precisions)) if precisions else 0.0


TOPK_DEFAULT = 4


def score_chunks(frame_ts: np.ndarray, frame_scores: np.ndarray,
                 chunks: np.ndarray, agg: str = "topk",
                 topk: int = TOPK_DEFAULT) -> np.ndarray:
    """Aggregate per-frame retrieval scores into per-chunk scores.

    The obvious choice is `max` -- "does this chunk contain a matching frame?" --
    and it is measurably the wrong one. `max` over the n frames in a chunk is an
    extreme-order statistic, so it rewards chunks with high *within-chunk
    variance* (ones spanning a cut, say) as readily as chunks that contain the
    answer. Averaging the top k is a lower-variance statistic of the same
    quantity and is worth +2.6 R@1 on MomentSeeker for no extra compute.

    The best k tracks the number of frames per chunk rather than a number of
    seconds -- roughly n/2.5 -- which is what identifies this as an estimator
    effect rather than a temporal-persistence one. See scripts/temporal_why.py.
    """
    idx = np.clip(np.searchsorted(chunks[:, 0], frame_ts, "right") - 1,
                  0, len(chunks) - 1)
    out = np.full(len(chunks), -np.inf, dtype=np.float32)
    if agg == "max":
        np.maximum.at(out, idx, frame_scores)
    elif agg == "mean":
        sums = np.zeros(len(chunks), np.float64)
        cnts = np.zeros(len(chunks), np.float64)
        np.add.at(sums, idx, frame_scores)
        np.add.at(cnts, idx, 1.0)
        out = np.where(cnts > 0, sums / np.maximum(cnts, 1), -np.inf).astype(np.float32)
    elif agg == "topk":
        order = np.argsort(idx, kind="stable")
        idx_s, sc_s = idx[order], frame_scores[order]
        bounds = np.searchsorted(idx_s, np.arange(len(chunks) + 1))
        for c in range(len(chunks)):
            v = sc_s[bounds[c]:bounds[c + 1]]
            if v.size:
                out[c] = np.sort(v)[-min(topk, v.size):].mean()
    else:
        raise ValueError(f"unknown aggregation {agg!r}")
    # chunks with no sampled frame (possible at the tail) rank last
    out[~np.isfinite(out)] = -1e9
    return out
