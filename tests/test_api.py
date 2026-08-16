"""The public API's contract.

These are the guarantees someone writing against `import framesieve` is entitled
to rely on, so they are tested separately from the algorithm tests in
test_core.py. Nothing here touches a GPU or a model: the parts that do are
exercised by the smoke test at the bottom, which skips when there is no video.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import framesieve as fs  # noqa: E402
from framesieve.api import Hit, SearchResults, VideoIndex, timecode  # noqa: E402
from framesieve.index import FrameIndex, IndexStats  # noqa: E402

DEMO = os.path.join(os.path.dirname(__file__), "..", "data", "demo_clip.mp4")


def test_importing_framesieve_does_not_import_torch():
    """The CLI's --help and any script that only reads an index should not pay
    for CUDA initialisation. This is the guarantee that keeps that true."""
    out = os.popen(
        f'{sys.executable} -c "'
        f"import sys; sys.path.insert(0, {os.path.join(os.path.dirname(__file__), '..', 'src')!r}); "
        f'import framesieve; print(\'torch\' in sys.modules)"'
    ).read().strip()
    assert out == "False", out


def test_public_names_are_all_reachable():
    for name in fs.__all__:
        assert getattr(fs, name) is not None, name
    assert fs.__version__


def test_unknown_attribute_raises_attribute_error():
    with pytest.raises(AttributeError):
        _ = fs.no_such_thing


def _fake_index(n=100, dim=8, fps=1.0) -> VideoIndex:
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(n, dim)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    stats = IndexStats(video="fake.mp4", duration_s=n / fps, target_fps=fps,
                       n_frames=n, n_encoded=n, n_segments=4,
                       encoder="test", encoder_revision="0", embed_dim=dim,
                       pixel_gate_tau=0.0, segment_tau=0.0, decode_encode_s=1.0,
                       frames_per_s=float(n), realtime_factor=1.0)
    fi = FrameIndex(np.arange(n, dtype=np.float32) / fps, emb,
                    (np.arange(n) // 25).astype(np.int32), stats)
    return VideoIndex(fi, video="fake.mp4")


def test_timecode_formats_hours_minutes_seconds():
    assert timecode(0) == "0:00:00"
    assert timecode(61.9) == "0:01:01"
    assert timecode(3725) == "1:02:05"
    assert timecode(-5) == "0:00:00"


def test_hit_confirmed_is_none_until_a_model_has_looked():
    assert Hit(1.0, 0.5).confirmed is None
    assert Hit(1.0, 0.5, vlm_score=0.1).confirmed is True
    assert Hit(1.0, 0.5, vlm_score=-0.1).confirmed is False


def test_video_index_exposes_shape_without_loading_a_model():
    v = _fake_index()
    assert len(v) == 100
    assert v.duration == pytest.approx(100.0)
    assert v.times.shape == (100,)
    assert v.embeddings.shape == (100, 8)
    # stored half, handed back as float32 so arithmetic on it behaves
    assert v.embeddings.dtype == np.float32
    assert "VideoIndex" in repr(v)


def test_search_results_slice_to_search_results_and_keep_order():
    hits = [Hit(float(i), 1.0 - 0.1 * i) for i in range(5)]
    r = SearchResults("q", hits, {"select_s": 0.001}, 5, "topk", False)
    assert len(r) == 5
    assert isinstance(r[0], Hit)
    assert isinstance(r[:2], SearchResults)
    assert len(r[:2]) == 2
    assert [h.time for h in r] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert r.times.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert r.latency_ms == pytest.approx(1.0)


def test_above_refuses_to_threshold_an_unconfirmed_similarity():
    """A retrieval score has no absolute scale, so thresholding it would be a
    number that means nothing. Better to refuse than to filter plausibly."""
    r = SearchResults("q", [Hit(0.0, 0.9)], {}, 1, "topk", confirmed=False)
    with pytest.raises(ValueError, match="confirm=True"):
        r.above(0.0)


def test_above_filters_on_the_vlm_score_when_there_is_one():
    hits = [Hit(0.0, 0.9, 2.0), Hit(1.0, 0.8, -1.0), Hit(2.0, 0.7, 0.5)]
    r = SearchResults("q", hits, {}, 3, "topk", confirmed=True)
    kept = r.above(0.0)
    assert [h.time for h in kept] == [0.0, 2.0]
    assert isinstance(kept, SearchResults)


def test_search_rejects_a_bad_strategy_and_a_bad_k():
    v = _fake_index()
    with pytest.raises(ValueError, match="unknown strategy"):
        v.search("x", strategy="nope")
    with pytest.raises(ValueError, match="at least 1"):
        v.search("x", k=0)


def test_index_path_encodes_the_encoder_and_rate():
    """Two indexes built with different encoders are not interchangeable, so the
    filename has to keep them apart."""
    a = fs.index_path_for("/tmp/a.mp4", "siglip2-base-224", 1.0)
    b = fs.index_path_for("/tmp/a.mp4", "siglip2-base-384", 1.0)
    c = fs.index_path_for("/tmp/a.mp4", "siglip2-base-224", 2.0)
    assert a != b != c and a != c
    assert all(p.endswith(".lance") and p.startswith("/tmp/a.") for p in (a, b, c))


def test_load_points_at_the_fix_when_there_is_no_index(tmp_path):
    with pytest.raises(FileNotFoundError, match="framesieve index"):
        fs.load(str(tmp_path / "missing.mp4"))


def test_frames_without_the_video_file_says_so():
    v = _fake_index()
    with pytest.raises(FileNotFoundError, match="frame store or the video"):
        v.frames([0.0, 1.0])
    assert v.frames([]) == []


def test_save_and_load_round_trips(tmp_path):
    v = _fake_index()
    p = v.save(str(tmp_path / "x.lance"))
    back = fs.load(p, video="fake.mp4")
    assert len(back) == len(v)
    assert np.allclose(back.times, v.times)
    assert np.allclose(back.embeddings, v.embeddings, atol=1e-3)


# --- the slow path, skipped unless the demo clip is present ------------------

pytestmark_video = pytest.mark.skipif(
    not os.path.exists(DEMO), reason="demo clip not downloaded")


@pytestmark_video
def test_end_to_end_index_search_and_score_agree(tmp_path):
    import shutil
    vid = str(tmp_path / "clip.mp4")
    shutil.copy(DEMO, vid)

    v = fs.index(vid, verbose=False)
    assert len(v) > 10
    assert os.path.exists(v.path)

    # open() must find what index() wrote rather than rebuilding it
    again = fs.open(vid)
    assert again.path == v.path
    assert len(again) == len(v)

    hits = v.search("a train", k=5)
    assert 0 < len(hits) <= 5
    assert hits.confirmed is False
    assert all(h.vlm_score is None for h in hits)
    # ranked best-first
    assert [h.score for h in hits] == sorted([h.score for h in hits], reverse=True)

    # the top hit's score must be the same number score() reports at that time
    curve = v.score("a train")
    i = int(np.argmin(np.abs(v.times - hits[0].time)))
    assert curve[i] == pytest.approx(hits[0].score, abs=1e-5)


def test_score_accepts_a_precomputed_vector_and_checks_its_shape():
    """The torch-free path: encode text on a GPU box, rank anywhere numpy runs."""
    v = _fake_index(dim=8)
    q = np.ones(8, dtype=np.float32)
    got = v.score(q)
    assert got.shape == (len(v),)
    assert np.allclose(got, v.embeddings @ (q / np.linalg.norm(q)))
    with pytest.raises(ValueError, match="8"):
        v.score(np.ones(7, dtype=np.float32))


def test_score_normalises_the_vector_it_is_given():
    """Scaling a query must not change what comes back.

    Asserted as identical ORDER plus a float32 tolerance, not as bit equality:
    normalising q and 17*q gives unit vectors that differ in the last bits, and
    the products then differ by ~1e-7 in a way that depends on the BLAS. An
    earlier version of this test used np.allclose's defaults, passed on aarch64
    and failed on x86.
    """
    v = _fake_index(dim=8)
    q = np.ones(8, dtype=np.float32)
    a, b = v.score(q), v.score(q * 17.0)
    assert np.array_equal(np.argsort(-a), np.argsort(-b))
    assert np.allclose(a, b, rtol=1e-4, atol=1e-6)


def test_device_selection_falls_back_rather_than_raising(monkeypatch):
    """No GPU is a normal machine, not an error.

    Before this, framesieve.index() on a CPU-only host died with
    `RuntimeError: No CUDA GPUs are available` from three frames down a torch
    stack, which tells the reader nothing about what to do.
    """
    import torch

    from framesieve.encoders import pick_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if getattr(torch.backends, "mps", None) is not None:
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert pick_device() == "cpu"
    assert pick_device("cuda") == "cuda"          # an explicit choice still wins

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert pick_device() == "cuda"


def test_cpu_gets_float32_because_bfloat16_is_emulated_there():
    import torch

    from framesieve.encoders import pick_dtype

    assert pick_dtype("cpu") is torch.float32
    assert pick_dtype("cuda") is torch.bfloat16
    assert pick_dtype("cpu", torch.float16) is torch.float16   # override wins


def test_index_path_is_one_lance_dataset():
    """Frames and vectors live in one Lance dataset -- the store is an extra
    column, not a second file -- so there is a single path per (video, encoder,
    rate) and nothing to disambiguate."""
    p = fs.index_path_for("/tmp/v.mp4")
    assert p == "/tmp/v.framesieve-siglip2-base-224-1fps.lance"


def test_importing_framesieve_does_not_pull_in_lance_or_lancedb():
    """Lance is the index format and a core dependency, but nothing should
    import it until an index is actually read or written -- and lancedb, which
    is optional, should not be imported at all until Collection is touched.

    Checked in a subprocess because sys.modules is process-wide and another
    test may legitimately have imported either.
    """
    import subprocess
    src = os.path.join(os.path.dirname(__file__), "..", "src")
    code = f"""
import sys
sys.path.insert(0, {src!r})
import numpy as np, framesieve as fs
from framesieve.index import FrameIndex, IndexStats
st = IndexStats(video="v.mp4", duration_s=4, target_fps=1, n_frames=4,
                n_encoded=4, n_segments=1, encoder="t", encoder_revision="0",
                embed_dim=4, pixel_gate_tau=0, segment_tau=0,
                decode_encode_s=1, frames_per_s=4, realtime_factor=1)
v = fs.VideoIndex(FrameIndex(np.arange(4, dtype=np.float32),
                             np.eye(4, dtype=np.float32),
                             np.zeros(4, np.int32), st))
v.score(np.ones(4, dtype=np.float32))
print(sorted(m for m in sys.modules if m.split(".")[0] in {{"lance", "lancedb"}}))
"""
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-800:]
    assert out.stdout.strip() == "[]", out.stdout


# --- Collection -------------------------------------------------------------

lancedb_missing = False
try:
    import lancedb  # noqa: F401
except ImportError:  # pragma: no cover - environment
    lancedb_missing = True

needs_lancedb = pytest.mark.skipif(lancedb_missing, reason="lancedb not installed")


def test_collection_is_reachable_but_not_imported_eagerly():
    """lancedb is an optional extra, so touching framesieve must not need it."""
    import subprocess
    src = os.path.join(os.path.dirname(__file__), "..", "src")
    out = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, {src!r}); import framesieve;"
         "print('lancedb' in sys.modules)"],
        capture_output=True, text=True)
    assert out.stdout.strip() == "False", out.stdout


@needs_lancedb
def test_collection_round_trips_and_collapses_runs(tmp_path):
    from framesieve.collection import Collection, CollectionHit

    lib = Collection(str(tmp_path / "c.lancedb"))
    assert len(lib) == 0 and lib.videos() == []

    rng = np.random.default_rng(0)
    for name in ("a.mp4", "b.mp4"):
        e = rng.normal(size=(50, 8)).astype(np.float32)
        e /= np.linalg.norm(e, axis=1, keepdims=True)
        lib._append(name, np.arange(50, dtype=np.float32), e)

    assert len(lib) == 100
    assert lib.videos() == ["a.mp4", "b.mp4"]

    q = rng.normal(size=8).astype(np.float32)
    hits = lib.search(q, k=5, exact=True, min_gap_s=0)
    assert len(hits) == 5
    assert all(isinstance(h, CollectionHit) for h in hits)
    # best first, and a similarity rather than a distance
    assert hits == sorted(hits, key=lambda h: -h.score)
    assert -1.01 <= hits[0].score <= 1.01


@needs_lancedb
def test_collapse_keeps_the_best_of_each_run():
    """Consecutive frames are near-identical, so an uncollapsed ranking returns
    one moment five times rather than five findings."""
    from framesieve.collection import Collection, CollectionHit

    hits = [CollectionHit("a.mp4", 100.0, 0.9), CollectionHit("a.mp4", 101.0, 0.89),
            CollectionHit("a.mp4", 102.0, 0.88), CollectionHit("b.mp4", 5.0, 0.80),
            CollectionHit("a.mp4", 900.0, 0.70)]
    kept = Collection._collapse(hits, k=5, min_gap_s=30.0, per_video=None)
    assert [(h.video, h.time) for h in kept] == [
        ("a.mp4", 100.0), ("b.mp4", 5.0), ("a.mp4", 900.0)]

    one_each = Collection._collapse(hits, k=5, min_gap_s=0, per_video=1)
    assert [h.video for h in one_each] == ["a.mp4", "b.mp4"]


@needs_lancedb
def test_reopening_a_collection_finds_what_is_already_in_it(tmp_path):
    """The create path and the reopen path are different code, and only the
    second one goes through table discovery -- which is where a lancedb API
    change silently made every existing collection look empty."""
    from framesieve.collection import Collection

    uri = str(tmp_path / "reopen.lancedb")
    rng = np.random.default_rng(1)
    e = rng.normal(size=(20, 8)).astype(np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True)

    first = Collection(uri)
    first._append("a.mp4", np.arange(20, dtype=np.float32), e)
    assert len(first) == 20

    again = Collection(uri)                 # a fresh handle, as a new process gets
    assert len(again) == 20, "reopened collection came back empty"
    assert again.videos() == ["a.mp4"]
    assert len(again.search(e[0], k=3, exact=True, min_gap_s=0)) == 3


def test_a_frameless_index_is_not_mistaken_for_a_frame_store(tmp_path):
    """Both forms are Lance datasets and only one carries the frames.

    FrameStore opens a frameless dataset perfectly happily and then fails much
    later, when someone asks for a frame -- so the decision has to be made on
    the schema, not on whether a constructor raises.
    """
    from framesieve.api import _has_frames

    v = _fake_index()
    p = v.save(str(tmp_path / "no_frames.lance"))
    assert _has_frames(p) is False
    assert _has_frames(str(tmp_path / "not_there.lance")) is False

    back = fs.load(p, video="fake.mp4")
    assert back._store is None, "a frameless index must not claim to be a store"
