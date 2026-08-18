"""Fast tests for the parts where a bug would be silent rather than loud.

Run: .venv/bin/python -m pytest tests -q

These use synthetic data and no models, so they take about a second. The checks
that need real weights, a real video and a GPU live in scripts/verify.py.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from framesieve.evaluate import (  # noqa: E402
    bootstrap_ci,
    evaluate_selection,
    event_recall,
    events_from_scores,
    frame_recall,
)
from framesieve.indexing import FrameIndex, IndexStats, _segment  # noqa: E402
from framesieve.search import STRATEGIES, select_candidates  # noqa: E402


def make_index(n=600, dim=16, seed=0, tau=0.0):
    """A synthetic index with real temporal structure: blocks of similar frames."""
    rng = np.random.default_rng(seed)
    blocks, emb = [], []
    i = 0
    while i < n:
        k = int(rng.integers(5, 40))
        base = rng.normal(size=dim)
        for _ in range(min(k, n - i)):
            v = base + 0.05 * rng.normal(size=dim)
            emb.append(v / np.linalg.norm(v))
        blocks.append(min(k, n - i))
        i += k
    emb = np.stack(emb[:n]).astype(np.float32)
    ts = np.arange(n, dtype=np.float32)
    seg = _segment(emb, tau)
    stats = IndexStats(video="synthetic", duration_s=float(n), target_fps=1.0,
                       n_frames=n, n_encoded=n,
                       n_segments=int(seg.max() + 1), encoder="synthetic",
                       encoder_revision="0", embed_dim=dim, pixel_gate_tau=0.0,
                       segment_tau=tau, decode_encode_s=1.0, frames_per_s=float(n),
                       realtime_factor=1.0)
    return FrameIndex(ts, emb, seg, stats)


# -- events ----------------------------------------------------------------


def test_events_merge_across_short_gaps():
    ts = np.arange(20, dtype=float)
    sc = np.full(20, -1.0)
    sc[[3, 4, 6, 7]] = 1.0          # one gap of 2 s inside the run
    ev = events_from_scores(ts, sc, merge_gap_s=3.0)
    assert len(ev) == 1 and ev[0].t_start == 3 and ev[0].t_end == 7


def test_events_split_across_long_gaps():
    ts = np.arange(20, dtype=float)
    sc = np.full(20, -1.0)
    sc[[3, 4, 15, 16]] = 1.0
    assert len(events_from_scores(ts, sc, merge_gap_s=3.0)) == 2


def test_no_events_when_nothing_positive():
    ts = np.arange(10, dtype=float)
    assert events_from_scores(ts, np.full(10, -1.0)) == []


# -- metrics ---------------------------------------------------------------


def test_event_recall_counts_events_not_frames():
    ts = np.arange(100, dtype=float)
    sc = np.full(100, -1.0)
    sc[10:50] = 1.0                 # one long event
    sc[80] = 1.0                    # one single-frame event
    ev = events_from_scores(ts, sc)
    assert len(ev) == 2
    # hitting the long event once and missing the short one is 50%, regardless
    # of how many of the 40 frames were found
    assert event_recall(ev, np.array([11.0])) == 0.5
    assert event_recall(ev, np.array([11.0, 12.0, 13.0])) == 0.5
    assert event_recall(ev, np.array([11.0, 80.0])) == 1.0


def test_frame_recall_is_capped_by_budget():
    ts = np.arange(100, dtype=float)
    gt_pos = ts[10:50]
    assert frame_recall(gt_pos, np.array([11.0])) == pytest.approx(1 / 40)


def test_evaluate_selection_requires_vlm_confirmation():
    ts = np.arange(100, dtype=float)
    sc = np.full(100, -1.0)
    sc[10:20] = 1.0
    ev = events_from_scores(ts, sc)
    # selecting a negative frame confirms nothing
    er, fr, pr, n = evaluate_selection(ts, sc, np.array([90.0]), ev)
    assert (er, n) == (0.0, 0)
    # selecting a positive frame does
    er, fr, pr, n = evaluate_selection(ts, sc, np.array([12.0]), ev)
    assert er == 1.0 and n == 1 and pr == 1.0
    # precision is over what was spent, not what was confirmed
    er, fr, pr, n = evaluate_selection(ts, sc, np.array([12.0, 90.0]), ev)
    assert pr == 0.5


def test_bootstrap_ci_brackets_the_mean():
    v = np.random.default_rng(0).normal(0.5, 0.1, 200)
    m, lo, hi = bootstrap_ci(v, n_boot=500)
    assert lo < m < hi and hi - lo < 0.1


# -- selection -------------------------------------------------------------


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("budget", [1, 4, 32, 200])
def test_selection_respects_budget_and_is_unique(strategy, budget):
    idx = make_index(tau=0.9)
    q = np.random.default_rng(1).normal(size=idx.emb.shape[1]).astype(np.float32)
    q /= np.linalg.norm(q)
    c = select_candidates(idx, q, budget, strategy=strategy)
    assert len(c.ts) <= budget, "spent more than the budget"
    assert len(set(c.ts.tolist())) == len(c.ts), "returned the same frame twice"
    assert np.isin(c.ts, idx.ts).all(), "returned a timestamp not in the index"


def test_segment_equals_topk_when_collapse_is_off():
    """The ablation's control: tau=0 must make `segment` degenerate to `topk`,
    otherwise the tau sweep is not isolating redundancy collapse."""
    idx = make_index(tau=0.0)
    q = np.random.default_rng(2).normal(size=idx.emb.shape[1]).astype(np.float32)
    for k in (1, 5, 25, 100):
        a = select_candidates(idx, q, k, strategy="segment")
        b = select_candidates(idx, q, k, strategy="topk")
        assert np.array_equal(np.sort(a.ts), np.sort(b.ts))


def test_uniform_is_seeded_and_seed_sensitive():
    idx = make_index()
    q = np.random.default_rng(3).normal(size=idx.emb.shape[1]).astype(np.float32)
    a = select_candidates(idx, q, 16, strategy="uniform", seed=0)
    b = select_candidates(idx, q, 16, strategy="uniform", seed=0)
    c = select_candidates(idx, q, 16, strategy="uniform", seed=1)
    assert np.array_equal(a.ts, b.ts)
    assert not np.array_equal(a.ts, c.ts)


def test_index_strategies_are_deterministic():
    idx = make_index(tau=0.9)
    q = np.random.default_rng(4).normal(size=idx.emb.shape[1]).astype(np.float32)
    for s in ("topk", "nms", "segment"):
        assert np.array_equal(select_candidates(idx, q, 16, strategy=s).ts,
                              select_candidates(idx, q, 16, strategy=s).ts)


def test_nms_window_adapts_so_a_large_budget_is_spendable():
    """A fixed NMS window silently caps how much budget it can spend, which shows
    up as a flat recall curve that looks like a finding."""
    idx = make_index(n=600, tau=0.9)
    q = np.random.default_rng(5).normal(size=idx.emb.shape[1]).astype(np.float32)
    adaptive = len(select_candidates(idx, q, 200, strategy="nms").ts)
    fixed = len(select_candidates(idx, q, 200, strategy="nms", nms_window_s=30.0).ts)
    assert adaptive > fixed
    assert adaptive >= 150


def test_segments_partition_the_frames():
    idx = make_index(tau=0.9)
    starts, ends, _, _ = idx.segments()
    assert starts[0] == 0 and ends[-1] == len(idx.ts)
    assert np.array_equal(starts[1:], ends[:-1])


def test_segment_reps_are_unit_norm():
    idx = make_index(tau=0.9)
    n = np.linalg.norm(idx.segment_reps(), axis=1)
    assert np.abs(n - 1).max() < 1e-5


# --- chunk aggregation ------------------------------------------------------
# The vectorised top-k path in score_chunks is the change that took the cheap
# stage past the published frontier, so it is worth pinning against a reference
# implementation rather than against its own output.

def test_topk_aggregation_matches_a_naive_reference():
    from framesieve.benchmarks.momentseeker import chunks_for, score_chunks
    rng = np.random.default_rng(0)
    ts = np.arange(0.0, 97.0, 1.0)
    scores = rng.normal(size=len(ts)).astype(np.float32)
    ch = chunks_for(97.0)
    for k in (1, 2, 4, 20):
        got = score_chunks(ts, scores, ch, agg="topk", topk=k)
        want = np.array([
            np.sort(scores[(ts >= a) & (ts < b)])[-k:].mean()
            if ((ts >= a) & (ts < b)).any() else -1e9
            for a, b in ch])
        assert np.allclose(got, want, atol=1e-5), k


def test_topk_at_k_one_is_exactly_max_and_at_large_k_is_the_mean():
    from framesieve.benchmarks.momentseeker import chunks_for, score_chunks
    rng = np.random.default_rng(1)
    ts = np.arange(0.0, 53.0, 1.0)
    s = rng.normal(size=len(ts)).astype(np.float32)
    ch = chunks_for(53.0)
    assert np.allclose(score_chunks(ts, s, ch, agg="topk", topk=1),
                       score_chunks(ts, s, ch, agg="max"), atol=1e-5)
    assert np.allclose(score_chunks(ts, s, ch, agg="topk", topk=10_000),
                       score_chunks(ts, s, ch, agg="mean"), atol=1e-5)


def test_chunks_with_no_sampled_frame_rank_last():
    from framesieve.benchmarks.momentseeker import chunks_for, score_chunks
    # a gap in the timestamps leaves the second chunk empty
    ts = np.array([0.0, 1.0, 2.0, 25.0, 26.0])
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    ch = chunks_for(30.0)
    out = score_chunks(ts, s, ch, agg="topk", topk=4)
    assert out[1] < -1e8
    assert np.argsort(-out)[-1] == 1


# --- pooling ----------------------------------------------------------------
# topk_mean is the module other people are most likely to copy, so its two
# endpoints and its ragged path are pinned explicitly.

def test_topk_mean_endpoints_are_max_and_mean():
    from framesieve.pooling import topk_mean
    rng = np.random.default_rng(3)
    x = rng.normal(size=(7, 11))
    assert np.allclose(topk_mean(x, 1), x.max(axis=-1))
    assert np.allclose(topk_mean(x, 11), x.mean(axis=-1))
    assert np.allclose(topk_mean(x, 99), x.mean(axis=-1))   # k clamps to n


def test_topk_mean_matches_a_sorted_reference_and_handles_ragged_rows():
    from framesieve.pooling import topk_mean
    rng = np.random.default_rng(4)
    x = rng.normal(size=(5, 9))
    for k in (2, 3, 5):
        want = np.sort(x, axis=-1)[:, -k:].mean(axis=-1)
        assert np.allclose(topk_mean(x, k), want), k
    ragged = [[1.0, 5.0, 3.0], [2.0, 2.0], [9.0]]
    got = topk_mean(ragged, 2)
    assert np.allclose(got, [4.0, 2.0, 9.0])


def test_estimate_m_uses_the_median_and_refuses_a_tiny_sample():
    from framesieve.pooling import estimate_m, recommend_k
    # right-skewed on purpose: one candidate matches everywhere
    # counts are six 1s, five 2s and one 4 -> median 1.5, mean 1.58; the median
    # is what keeps the single all-matching candidate from moving the answer
    rel = [[1, 0, 0, 0]] * 6 + [[1, 1, 0, 0]] * 5 + [[1] * 4]
    assert estimate_m(rel) == 1.5
    # recommend_k halves the counted m -- calibrated in COUNTED_M_TO_K against
    # five measured datasets -- and never returns less than 1
    assert recommend_k(rel) == 1
    try:
        estimate_m([[1, 0]], min_examples=10)
    except ValueError as e:
        assert "at least 10" in str(e)
    else:
        raise AssertionError("expected a refusal on too few examples")


def test_recommend_k_halves_the_count_and_k_range_stops_at_it():
    from framesieve.pooling import k_range, recommend_k
    rel = [[1] * 8 + [0] * 2] * 20          # counted m = 8, as on MomentSeeker
    assert recommend_k(rel) == 4            # which is the measured optimum there
    ks = k_range(rel)
    assert ks[0] == 1 and max(ks) == 8


def test_sweep_k_finds_the_planted_optimum_under_a_head_metric():
    from framesieve.pooling import sweep_k
    rng = np.random.default_rng(5)
    n, m, n_neg = 32, 8, 49
    scores, pos, grp = [], [], []
    for q in range(300):
        c = rng.normal(0.0, 1.0, size=n)
        c[:m] = rng.normal(1.6, 1.0, size=m)
        scores.append(c)
        pos.append(True)
        grp.append(q)
        for _ in range(n_neg):
            scores.append(rng.normal(0.0, 1.0, size=n))
            pos.append(False)
            grp.append(q)
    r = sweep_k(scores, pos, grp, n_boot=200)
    # planted m is 8; the measured optimum sits near it and well off both ends
    assert 4 <= r["best_k"] <= 16, r["best_k"]
    assert r["by_k"][r["best_k"]]["score"] > r["by_k"][1]["score"]
    assert r["by_k"][r["best_k"]]["score"] > r["by_k"][32]["score"]


# --- regressions from the launch review ---------------------------------------


def test_streaming_segmenter_matches_the_batch_segmentation():
    """The frame store writes batches to disk as they stream past, so its
    segmentation is computed incrementally. Any drift from `_segment` would
    silently change which frames `strategy="segment"` and the OCR pass pick."""
    from framesieve.indexing import StreamingSegmenter

    rng = np.random.default_rng(3)
    emb = rng.normal(size=(500, 16)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    # a few plateaus so tau actually merges something
    for s in (50, 200, 400):
        emb[s:s + 30] = emb[s]

    for tau in (0.0, 0.3, 0.9, 0.99):
        want = _segment(emb, tau)
        seg = StreamingSegmenter(tau)
        got = np.concatenate([seg.feed(emb[i:i + 64])
                              for i in range(0, len(emb), 64)])
        assert np.array_equal(got, want), tau
        assert seg.n_segments == (int(want.max()) + 1 if len(want) else 0), tau


def test_gpu_decode_picks_the_decoder_for_the_actual_codec():
    """gpu=True used to hardcode h264_cuvid and fail on HEVC/VP9/AV1 with
    'no frames decoded', with the probed codec in hand two lines earlier."""
    from framesieve.frames import _cuvid_decoder

    assert _cuvid_decoder("h264") == "h264_cuvid"
    assert _cuvid_decoder("hevc") == "hevc_cuvid"
    assert _cuvid_decoder("av1") == "av1_cuvid"
    with pytest.raises(ValueError, match="gpu_decode"):
        _cuvid_decoder("theora")
