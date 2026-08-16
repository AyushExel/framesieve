"""Recall vs compute, measured against the dense-VLM ground truth.

The unit of cost is one VLM call, because on the measured hardware it is 246-809x
anything else in the pipeline. The unit of accuracy is *event* recall, not frame
recall, and that choice matters:

  frame recall  penalises you for not finding all 40 frames of a 40 s event, which
                nobody cares about, and is capped at budget/|positives| whenever
                the event is long. It makes every method look bad for the wrong
                reason.
  event recall  asks the question a user actually asks: for each thing that
                happened, did we surface it at all?

Both are reported. Event recall is the headline.

Evaluation looks up ground-truth scores at the selected timestamps rather than
re-running the VLM. That is exact -- same model, same frames, same settings -- and
it is what makes a 4-strategy x 12-budget x 20-seed sweep affordable. The
assumption that lookup equals recompute is not assumed; it is tested in
scripts/verify.py.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------
# ground truth -> events
# --------------------------------------------------------------------------


@dataclass
class Event:
    t_start: float
    t_end: float
    n_frames: int
    peak_score: float

    @property
    def duration_s(self) -> float:
        return self.t_end - self.t_start


def events_from_scores(ts: np.ndarray, scores: np.ndarray, *, threshold: float = 0.0,
                       merge_gap_s: float = 3.0, min_frames: int = 1) -> list[Event]:
    """Contiguous runs of positive frames, merged across short gaps.

    merge_gap_s exists because a real event flickers: a tunnel mouth briefly
    occluded for one frame is still one tunnel, and counting it as two events
    would flatter any method that happens to hit the middle.
    """
    pos = np.flatnonzero(scores > threshold)
    if len(pos) == 0:
        return []
    groups: list[list[int]] = [[pos[0]]]
    for i in pos[1:]:
        if ts[i] - ts[groups[-1][-1]] <= merge_gap_s:
            groups[-1].append(i)
        else:
            groups.append([i])
    out = []
    for g in groups:
        if len(g) < min_frames:
            continue
        out.append(Event(t_start=float(ts[g[0]]), t_end=float(ts[g[-1]]),
                         n_frames=len(g), peak_score=float(scores[g].max())))
    return out


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def event_recall(events: Sequence[Event], hit_ts: np.ndarray,
                 tol_s: float = 0.0) -> float:
    """Fraction of events with at least one confirmed frame inside them."""
    if not events:
        return float("nan")
    if len(hit_ts) == 0:
        return 0.0
    h = np.sort(hit_ts)
    found = 0
    for e in events:
        lo = np.searchsorted(h, e.t_start - tol_s, "left")
        hi = np.searchsorted(h, e.t_end + tol_s, "right")
        found += int(hi > lo)
    return found / len(events)


def frame_recall(gt_pos_ts: np.ndarray, hit_ts: np.ndarray) -> float:
    if len(gt_pos_ts) == 0:
        return float("nan")
    return float(np.isin(gt_pos_ts, hit_ts).sum() / len(gt_pos_ts))


def precision(hit_ts: np.ndarray, budget: int) -> float:
    return float(len(hit_ts) / budget) if budget else 0.0


# --------------------------------------------------------------------------
# the sweep
# --------------------------------------------------------------------------


@dataclass
class EvalRow:
    query: str
    strategy: str
    budget: int
    seed: int
    n_events: int
    n_gt_positive: int
    event_recall: float
    frame_recall: float
    precision: float
    n_selected: int
    n_confirmed: int


def evaluate_selection(gt_ts: np.ndarray, gt_scores: np.ndarray,
                       selected_ts: np.ndarray, events: Sequence[Event],
                       *, threshold: float = 0.0,
                       max_snap_s: float = 1.5) -> tuple[float, float, float, int]:
    """Score one candidate set against ground truth.

    A selected frame counts as a hit only if the VLM confirms it -- which, because
    ground truth is that same VLM on that same frame, is a lookup. This is what
    makes the cascade's precision meaningful: retrieval proposes, the VLM decides.
    """
    if len(selected_ts) == 0:
        return 0.0, 0.0, 0.0, 0
    idx = np.searchsorted(gt_ts, selected_ts)
    idx = np.clip(idx, 0, len(gt_ts) - 1)
    # snap to nearest ground-truth frame (selection is on the same 1 fps grid,
    # but guard against float drift)
    left = np.clip(idx - 1, 0, len(gt_ts) - 1)
    take_left = np.abs(gt_ts[left] - selected_ts) < np.abs(gt_ts[idx] - selected_ts)
    idx = np.where(take_left, left, idx)

    # A selection outside the ground truth's time range would otherwise be
    # snapped onto the first or last covered frame and scored as if it had been
    # there -- silently, and wrongly. This happens whenever the index covers more
    # of the video than the ground truth does, e.g. against a partial run.
    far = np.abs(gt_ts[idx] - selected_ts) > max_snap_s
    if far.any():
        raise ValueError(
            f"{int(far.sum())}/{len(selected_ts)} selected timestamps are more "
            f"than {max_snap_s}s from any ground-truth frame (worst "
            f"{np.abs(gt_ts[idx] - selected_ts).max():.1f}s). Restrict the index "
            "to the ground truth's time range before evaluating.")

    confirmed = idx[gt_scores[idx] > threshold]
    hit_ts = gt_ts[confirmed]
    gt_pos_ts = gt_ts[gt_scores > threshold]
    return (event_recall(events, hit_ts), frame_recall(gt_pos_ts, hit_ts),
            precision(hit_ts, len(selected_ts)), int(len(confirmed)))


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    """Mean and percentile CI, resampling over whatever axis was handed in.

    Used to put error bars on comparisons where the strategies themselves are
    deterministic: the spread that matters there is across queries, not seeds.
    """
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return float(v.mean()), float(np.percentile(boots, 100 * alpha / 2)), \
        float(np.percentile(boots, 100 * (1 - alpha / 2)))
