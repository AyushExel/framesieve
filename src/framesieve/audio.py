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

Needs `pip install "framesieve[audio]"`.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass

import numpy as np

__all__ = ["SpeechSegment", "SpeechIndex", "DEFAULT_ASR", "DEFAULT_TEXT_ENCODER",
           "transcribe", "build_speech_index", "speech_path_for", "has_audio"]

DEFAULT_ASR = "openai/whisper-small"
DEFAULT_TEXT_ENCODER = "BAAI/bge-small-en-v1.5"

# bge wants the query side marked and the passage side bare; using the wrong one
# costs a couple of points and would look like a retrieval problem
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass(frozen=True)
class SpeechSegment:
    start: float
    end: float
    text: str


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
               verbose: bool = False) -> list[SpeechSegment]:
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

    segs: list[SpeechSegment] = []
    for c in out.get("chunks") or []:
        ts = c.get("timestamp") or (None, None)
        start, end = ts[0], ts[1]
        text = (c.get("text") or "").strip()
        # the last chunk can come back with an open end, and empty text is what
        # Whisper emits for silence -- neither is worth indexing
        if start is None or not text:
            continue
        segs.append(SpeechSegment(float(start),
                                  float(end if end is not None else start + chunk_s),
                                  text))
    del asr
    if device == "cuda":
        torch.cuda.empty_cache()
    return segs


class SpeechIndex:
    """Timed transcript segments and their embeddings."""

    def __init__(self, segments: list[SpeechSegment], emb: np.ndarray,
                 meta: dict | None = None):
        self.segments = segments
        self.emb = np.asarray(emb, dtype=np.float32)
        self.meta = meta or {}

    def __len__(self) -> int:
        return len(self.segments)

    def __repr__(self) -> str:
        mins = (self.segments[-1].end / 60) if self.segments else 0.0
        return f"<SpeechIndex {len(self)} segments over {mins:.1f} min>"

    @property
    def starts(self) -> np.ndarray:
        return np.array([s.start for s in self.segments], dtype=np.float64)

    def save(self, path: str) -> None:
        import lance
        import pyarrow as pa

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        d = self.emb.shape[1] if len(self.emb) else 0
        table = pa.table({
            "start": pa.array([s.start for s in self.segments], pa.float64()),
            "end": pa.array([s.end for s in self.segments], pa.float64()),
            "text": pa.array([s.text for s in self.segments], pa.string()),
            "emb": pa.FixedSizeListArray.from_arrays(
                pa.array(np.ascontiguousarray(self.emb).reshape(-1)), d),
        })
        lance.write_dataset(table, path, mode="overwrite")
        with open(os.path.join(path, "framesieve.json"), "w") as f:
            json.dump({"meta": self.meta}, f)

    @classmethod
    def load(cls, path: str) -> SpeechIndex:
        import lance

        t = lance.dataset(path).to_table(columns=["start", "end", "text", "emb"])
        segs = [SpeechSegment(float(a), float(b), c) for a, b, c in
                zip(t.column("start").to_pylist(), t.column("end").to_pylist(),
                    t.column("text").to_pylist())]
        emb = (np.stack(t.column("emb").to_numpy(zero_copy_only=False))
               if len(segs) else np.zeros((0, 0), np.float32))
        with open(os.path.join(path, "framesieve.json")) as f:
            meta = json.load(f).get("meta", {})
        return cls(segs, emb, meta)


class TextEncoder:
    """The sentence encoder used for speech, on both sides of the match."""

    def __init__(self, name: str = DEFAULT_TEXT_ENCODER,
                 device: str | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        from .encoders import pick_device, pick_dtype

        self.device = pick_device(device)
        self.dtype = pick_dtype(self.device, None)
        self.name = name
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(
            name, torch_dtype=self.dtype).to(self.device).eval()
        self._torch = torch

    def encode(self, texts, *, query: bool = False, batch: int = 128,
               maxlen: int = 512) -> np.ndarray:
        torch = self._torch
        prefix = BGE_QUERY_PREFIX if query else ""
        out = []
        for i in range(0, len(texts), batch):
            b = [prefix + t for t in texts[i:i + batch]]
            enc = self.tok(b, padding=True, truncation=True, max_length=maxlen,
                           return_tensors="pt").to(self.device)
            with torch.inference_mode():
                h = self.model(**enc).last_hidden_state[:, 0]   # bge uses CLS
                h = torch.nn.functional.normalize(h.float(), dim=-1)
            out.append(h.cpu().numpy().astype(np.float32))
        return (np.concatenate(out) if out
                else np.zeros((0, self.model.config.hidden_size), np.float32))


def build_speech_index(video: str, *, asr: str = DEFAULT_ASR,
                       text_encoder: str = DEFAULT_TEXT_ENCODER,
                       device: str | None = None, language: str | None = None,
                       verbose: bool = False) -> SpeechIndex:
    """Transcribe a video and embed its segments."""
    segs = transcribe(video, model=asr, device=device, language=language,
                      verbose=verbose)
    enc = TextEncoder(text_encoder, device=device)
    emb = enc.encode([s.text for s in segs])
    del enc
    return SpeechIndex(segs, emb, meta={"video": video, "asr": asr,
                                        "text_encoder": text_encoder,
                                        "n_segments": len(segs)})
