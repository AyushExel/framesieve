"""Reading the text that is on screen.

A 224-pixel global embedding cannot read. On MomentSeeker's OCR split the
retrieval stage scores R@1 3.4, which is close to chance -- a sign, a caption, a
slide title or a scoreboard is effectively invisible to it. This runs an OCR pass
and indexes what it finds, so those become searchable like anything else.

It is the most expensive index-time pass here: about 120 ms a frame, so roughly
7 minutes per hour of video against 5.5 for speech and 15 seconds for the frame
embeddings. Two things make that bearable.

    one frame per segment    Consecutive frames at 1 fps are usually the same
                             shot, and the index already knows which -- the
                             redundancy collapse that `segment_tau` performs is
                             exactly the grouping OCR wants. Reading one frame
                             per segment cut the pass by 4x on the test video.
                             It is the default, and `every="frame"` turns it off
                             for footage whose text changes under a still
                             picture, like a ticker or a live caption.

    confidence and length    OCR on video frames returns a lot of nothing:
                             single characters off a logo, fragments from motion
                             blur. Anything below `min_conf` or shorter than
                             `min_chars` is dropped rather than indexed and
                             later retrieved.

Needs `pip install "framesieve[ocr]"`.
"""

from __future__ import annotations

import os

import numpy as np

from .timedtext import DEFAULT_TEXT_ENCODER, TimedText, embed_segments

__all__ = ["read_frames", "build_text_index", "text_path_for", "DEFAULT_OCR"]

DEFAULT_OCR = "rapidocr"


def text_path_for(video: str) -> str:
    """Where the on-screen-text index for this video lives."""
    stem = os.path.splitext(video)[0]
    return f"{stem}.framesieve-{DEFAULT_OCR}.text.lance"


def _engine():
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:  # pragma: no cover - environment
        raise ImportError(
            'OCR needs an engine: pip install "framesieve[ocr]"\n'
            "  Everything else in framesieve works without it."
        ) from exc
    return RapidOCR()


def _read_one(ocr, frame: np.ndarray, min_conf: float, min_chars: int) -> str:
    """One frame's legible text, or the empty string.

    Frames are uint8 RGB, as everything else in framesieve produces them; the
    engine wants BGR, which is the [::-1] on the last axis.
    """
    res, _ = ocr(np.ascontiguousarray(frame[:, :, ::-1]))
    parts = [t for _, t, conf in (res or [])
             if conf >= min_conf and len(t.strip()) >= min_chars]
    return " ".join(p.strip() for p in parts)


def read_frames(frames, *, min_conf: float = 0.5, min_chars: int = 3,
                verbose: bool = False) -> list[str]:
    """One line of text per frame, empty where nothing legible was found."""
    ocr = _engine()
    out: list[str] = []
    for i, f in enumerate(frames):
        out.append(_read_one(ocr, f, min_conf, min_chars))
        if verbose and (i + 1) % 200 == 0:
            print(f"    {i + 1} frames read", flush=True)
    return out


def build_text_index(video: str, *, index=None, fps: float = 1.0,
                     every: str = "segment", size: int = 640,
                     min_conf: float = 0.5, min_chars: int = 3,
                     text_encoder: str = DEFAULT_TEXT_ENCODER,
                     device: str | None = None, verbose: bool = False):
    """Read the on-screen text of a video and index it.

    index   the FrameIndex for this video, when there is one. It supplies the
            segmentation that `every="segment"` uses, and the timestamps, so the
            OCR pass looks at the same frames the retrieval stage did.
    every   "segment" reads one frame per shot, "frame" reads all of them
    size    decode height for OCR. Text needs more pixels than the 224 the
            retrieval encoder wants, and less than native: 640 was enough to
            read captions on the test footage
    """
    from .frames import FrameStream, probe_source

    if every not in ("segment", "frame"):
        raise ValueError(f"every must be 'segment' or 'frame', got {every!r}")

    info = probe_source(video)
    # `size` is the decode HEIGHT. Handing an int straight to FrameStream would
    # decode squares -- 1920x1080 squashed to 640x640 -- which distorts exactly
    # the text this pass exists to read. Preserve the aspect ratio (ffmpeg wants
    # even dimensions).
    if info.height > 0 and info.width > 0:
        w = max(2, 2 * round(info.width * size / info.height / 2))
    else:  # pragma: no cover - a broken probe; squares beat crashing
        w = size
    stream = FrameStream(video, target_fps=fps, size=(w, size), batch=64)

    # which sampled frames to actually read. With a segmentation, one per
    # segment: consecutive frames of one shot carry the same text, and reading
    # all of them is the same answer four times over.
    wanted: set[int] | None = None
    if index is not None and every == "segment":
        starts, _, _, _ = index.segments()
        wanted = set(int(i) for i in starts)
        if verbose:
            print(f"  {len(wanted)} segments out of {len(index.ts)} frames")

    # read as the frames stream past, keeping the text and dropping the pixels:
    # buffering every selected frame first held gigabytes for a long video
    ocr = _engine()
    times: list[float] = []
    texts: list[str] = []
    n = n_read = 0
    for ts, batch in stream:
        for t, fr in zip(ts, batch):
            if wanted is None or n in wanted:
                times.append(float(t))
                texts.append(_read_one(ocr, fr, min_conf, min_chars))
                n_read += 1
                if verbose and n_read % 200 == 0:
                    print(f"    {n_read} frames read", flush=True)
            n += 1
    if verbose:
        print(f"  read {n_read} frames at {size}px")

    # a span runs until the next frame that was read, so a shot's text covers
    # the shot rather than a single instant
    segs: list[TimedText] = []
    for i, (t, txt) in enumerate(zip(times, texts)):
        if not txt:
            continue
        end = times[i + 1] if i + 1 < len(times) else min(
            t + 1.0 / fps, info.duration_s)
        segs.append(TimedText(t, float(end), txt))
    if verbose:
        print(f"  {len(segs)} frames carried legible text")

    return embed_segments(segs, "text", text_encoder=text_encoder,
                          device=device,
                          meta={"video": video, "ocr": DEFAULT_OCR,
                                "every": every, "read": n_read})
