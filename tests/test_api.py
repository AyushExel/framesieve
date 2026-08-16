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


def test_index_path_names_the_store_form_separately():
    """A frame store and a plain index are different files, so `--store` cannot
    silently overwrite a plain index or be mistaken for one."""
    npz = fs.index_path_for("/tmp/a.mp4", "siglip2-base-224", 1.0)
    lance = fs.index_path_for("/tmp/a.mp4", "siglip2-base-224", 1.0, store=True)
    assert npz.endswith(".npz") and lance.endswith(".lance")
    assert npz[: -len(".npz")] == lance[: -len(".lance")]


def test_index_path_encodes_the_encoder_and_rate():
    """Two indexes built with different encoders are not interchangeable, so the
    filename has to keep them apart."""
    a = fs.index_path_for("/tmp/a.mp4", "siglip2-base-224", 1.0)
    b = fs.index_path_for("/tmp/a.mp4", "siglip2-base-384", 1.0)
    c = fs.index_path_for("/tmp/a.mp4", "siglip2-base-224", 2.0)
    assert a != b != c and a != c
    assert all(p.endswith(".npz") and p.startswith("/tmp/a.") for p in (a, b, c))


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
    p = v.save(str(tmp_path / "x.npz"))
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
    v = _fake_index(dim=8)
    q = np.ones(8, dtype=np.float32)
    assert np.allclose(v.score(q), v.score(q * 17.0))


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


def test_store_and_plain_index_are_different_files():
    """`--store` must not be able to overwrite a plain index, or be loaded as
    one: they hold different things and the loader branches on the suffix."""
    a = fs.index_path_for("/tmp/v.mp4", store=False)
    b = fs.index_path_for("/tmp/v.mp4", store=True)
    assert a != b
    assert a.endswith(".npz") and b.endswith(".lance")
