"""What was said, indexed alongside what was shown.

A frame encoder cannot hear. On a recorded meeting, a lecture, an interview or
most of YouTube, the thing you want to find was spoken rather than displayed --
and no amount of visual retrieval will reach it.

This transcribes the audio with Whisper, keeps the timed segments, and embeds
each one with a text encoder so they can be searched the same way frames are.

Two encoders are in play and that is deliberate. Frames are matched with SigLIP,
whose text tower is trained to sit beside images and is a poor text-to-text
matcher; speech is matched with a sentence encoder, which is what that job wants.
The cost is a second small forward pass per query, about 5 ms.

Scores from the two therefore live on different scales and are NOT comparable:
a 0.16 against a frame and a 0.16 against a sentence mean different things. The
search layer keeps them ranked within their own modality and labels every hit
with which one it came from, rather than pretending one number orders both.

    framesieve index video.mp4 --audio
    framesieve search video.mp4 "the part about pricing"

Needs `pip install "framesieve[audio]"`. The container and the sentence encoder
are shared with the OCR pass, in `framesieve.timedtext` -- speech and on-screen
text arrive by different routes and are the same thing once they arrive.
"""

from __future__ import annotations

import os
import subprocess

import numpy as np

from .timedtext import DEFAULT_TEXT_ENCODER, TimedText, embed_segments

__all__ = ["DEFAULT_ASR", "transcribe", "build_speech_index",
           "speech_path_for", "has_audio"]

DEFAULT_ASR = "openai/whisper-small"


def speech_path_for(video: str, asr: str = DEFAULT_ASR) -> str:
    """Where the speech index for this video lives.

    A sibling of the frame index rather than a table inside it: the two are
    written by different passes, at different times, and either can exist
    without the other.
    """
    stem = os.path.splitext(video)[0]
    tag = asr.split("/")[-1]
    return f"{stem}.framesieve-{tag}.speech.lance"


def has_audio(video: str) -> bool:
    """Is there an audio stream at all? Silent footage is common, and asking
    Whisper to transcribe nothing wastes minutes and returns hallucinations."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_name", "-of", "csv=p=0", video],
            capture_output=True, text=True, timeout=30)
        return bool(out.stdout.strip())
    except Exception:
        return False


def extract_audio(video: str, sr: int = 16000) -> np.ndarray:
    """Mono float32 at 16 kHz, straight from ffmpeg, which is already required.

    Piped rather than written to a temp file: an hour of 16 kHz mono is 115 MB
    and there is no reason for it to touch disk.
    """
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-i", video,
           "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    out = subprocess.run(cmd, capture_output=True)
    if out.returncode != 0 or not out.stdout:
        raise RuntimeError(
            f"could not read audio from {video}: "
            f"{out.stderr.decode()[-300:] or 'no audio stream'}")
    return np.frombuffer(out.stdout, dtype=np.float32)


def transcribe(video: str, *, model: str = DEFAULT_ASR,
               device: str | None = None, language: str | None = None,
               chunk_s: float = 30.0, batch: int = 8,
               verbose: bool = False) -> list[TimedText]:
    """Timed transcript segments for a video.

    Whisper is trained on 30-second windows, so that is the chunk length; the
    pipeline stitches them and returns timestamps per segment rather than per
    word, which is the granularity a search over speech actually wants.
    """
    import torch
    from transformers import pipeline

    from .encoders import pick_device, pick_dtype

    device = pick_device(device)
    dtype = pick_dtype(device, None)
    audio = extract_audio(video)
    if verbose:
        print(f"  {len(audio) / 16000 / 60:.1f} min of audio -> {model}")

    asr = pipeline("automatic-speech-recognition", model=model,
                   torch_dtype=dtype, device=device,
                   chunk_length_s=chunk_s, batch_size=batch)
    kw = {"language": language} if language else {}
    out = asr(audio, return_timestamps=True, generate_kwargs=kw)

    segs: list[TimedText] = []
    for c in out.get("chunks") or []:
        ts = c.get("timestamp") or (None, None)
        start, end = ts[0], ts[1]
        text = (c.get("text") or "").strip()
        # the last chunk can come back with an open end, and empty text is what
        # Whisper emits for silence -- neither is worth indexing
        if start is None or not text:
            continue
        segs.append(TimedText(
            float(start),
            float(end if end is not None else start + chunk_s), text))
    del asr
    if device == "cuda":
        torch.cuda.empty_cache()
    return segs


def build_speech_index(video: str, *, asr: str = DEFAULT_ASR,
                       text_encoder: str = DEFAULT_TEXT_ENCODER,
                       device: str | None = None, language: str | None = None,
                       verbose: bool = False):
    """Transcribe a video and embed its segments."""
    segs = transcribe(video, model=asr, device=device, language=language,
                      verbose=verbose)
    return embed_segments(segs, "speech", text_encoder=text_encoder,
                          device=device, meta={"video": video, "asr": asr})
