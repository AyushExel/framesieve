"""One Lance dataset per video: embeddings, segmentation, and the frames themselves.

The refine stage needs arbitrary frames at full resolution, and seeking a video
for them is surprisingly expensive -- 107 ms per frame single-threaded, which is
more than the VLM call it feeds. Parallel seeking hides most of that, but it
burns 32 CPU workers to do it.

The observation that fixes it: **the indexer has already decoded every frame it
embeds.** Encoding each one as a JPEG on the way past and storing it in a blob
column turns "fetch the frame at t" from a seek-and-decode into a byte-range read.

Measured on the demo clip (bench/blobstore_bench.py), 32 random frames:

    ffmpeg seek, 1 worker      107.5 ms/frame
    ffmpeg seek, 32 workers     12.2 ms/frame
    lance blob, 8 decode thr     0.86 ms/frame     <- 14x, with 4x fewer threads
      of which byte-range read   0.09 ms/frame

and it costs *less* disk than the source video (0.29x), because one JPEG per
second is smaller than 25 H.264 frames per second.

Storing the embeddings in the same dataset means one artifact per video, and it
puts the vectors somewhere a real vector index can reach when the corpus grows
past what a brute-force matmul should handle.
"""

from __future__ import annotations

import io
import json
import os
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import numpy as np
import pyarrow as pa

from .encoders import SiglipEncoder
from .frames import FrameStream, probe_source
from .indexing import INDEX_FORMAT, FrameIndex, IndexStats, StreamingSegmenter

BLOB_META = {"lance-encoding:blob": "true"}
DEFAULT_JPEG_QUALITY = 90


def _require_lance():
    try:
        import lance
        return lance
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "the Lance frame store needs `pip install pylance`; "
            "framesieve works without it by seeking the video instead") from e


def _encode_jpeg(arr: np.ndarray, quality: int, max_pixels: int = 0) -> bytes:
    from PIL import Image
    img = Image.fromarray(arr)
    if max_pixels and img.width * img.height > max_pixels:
        s = (max_pixels / (img.width * img.height)) ** 0.5
        img = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))),
                         Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _decode_jpeg(raw: bytes) -> np.ndarray:
    from PIL import Image
    return np.array(Image.open(io.BytesIO(raw)).convert("RGB"))


def schema_for(dim: int) -> pa.Schema:
    return pa.schema([
        pa.field("frame_idx", pa.int32()),
        pa.field("ts", pa.float64()),
        pa.field("seg_id", pa.int32()),
        pa.field("emb", pa.list_(pa.float32(), dim)),
        pa.field("jpeg", pa.large_binary(), metadata=BLOB_META),
    ])


# --------------------------------------------------------------------------


def build_store(video: str, encoder: SiglipEncoder, out: str, *,
                target_fps: float = 1.0, size: int = 256, batch: int = 256,
                segment_tau: float = 0.90, jpeg_quality: int = DEFAULT_JPEG_QUALITY,
                store_max_pixels: int = 0, jpeg_workers: int = 16,
                gpu_decode: bool = False, seed: int = 0, start_s: float = 0.0,
                duration_s: float = 0.0, verbose: bool = True) -> dict:
    """Single pass: decode -> embed -> JPEG -> Lance.

    The frames are already in memory for the encoder, so the only extra cost is
    the JPEG encode, which is parallelised across `jpeg_workers` threads because
    PIL releases the GIL while it works.

    Batches stream straight into the dataset as they are produced. Holding them
    all and writing once would keep every JPEG in memory -- about 2.7 GB per
    hour of footage, which is an OOM on exactly the long videos a store is for.
    The segmentation that used to force that buffering is computed incrementally
    (`StreamingSegmenter` matches `_segment` batch-for-batch, pinned by a test).
    """
    lance = _require_lance()
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    info = probe_source(video)
    stream = FrameStream(video, target_fps=target_fps, size=None, batch=batch,
                         start_s=start_s, duration_s=duration_s, gpu=gpu_decode)

    schema = schema_for(encoder.spec.dim)
    seg = StreamingSegmenter(segment_tau)
    state = {"n": 0, "t_encode": 0.0, "t_jpeg": 0.0}
    t0 = time.perf_counter()

    def _batches():
        with ThreadPoolExecutor(jpeg_workers) as pool:
            for ts, frames in stream:
                t = time.perf_counter()
                # the encoder wants square model-resolution input; the store
                # wants the frame as shot, so resize only for the encoder
                small = np.stack([np.array(_resize(f, size)) for f in frames])
                emb = encoder.encode_frames(
                    torch.from_numpy(small)).cpu().numpy().astype(np.float32)
                state["t_encode"] += time.perf_counter() - t

                t = time.perf_counter()
                jp = list(pool.map(lambda f: _encode_jpeg(f, jpeg_quality,
                                                          store_max_pixels),
                                   frames))
                state["t_jpeg"] += time.perf_counter() - t

                n0 = state["n"]
                yield pa.record_batch(
                    [pa.array(np.arange(n0, n0 + len(frames)), pa.int32()),
                     pa.array(ts, pa.float64()),
                     pa.array(seg.feed(emb), pa.int32()),
                     pa.array(list(emb), pa.list_(pa.float32(), encoder.spec.dim)),
                     pa.array(jp, pa.large_binary())], schema=schema)
                state["n"] += len(frames)

    if os.path.exists(out):
        import shutil
        shutil.rmtree(out)
    lance.write_dataset(pa.RecordBatchReader.from_batches(schema, _batches()),
                        out, mode="create", data_storage_version="stable")
    wall = time.perf_counter() - t0
    n, t_encode, t_jpeg = state["n"], state["t_encode"], state["t_jpeg"]
    # decode and write interleave now, so this is "everything that is not the
    # encoder or the JPEG pass" rather than a pure write time
    t_write = max(0.0, wall - t_encode - t_jpeg)

    size_bytes = sum(os.path.getsize(os.path.join(dp, f))
                     for dp, _, fs in os.walk(out) for f in fs)
    covered = duration_s or max(0.0, info.duration_s - start_s)
    stats = IndexStats(
        video=os.path.abspath(video), duration_s=covered,
        target_fps=target_fps, n_frames=n, n_encoded=n,
        n_segments=seg.n_segments if n else 0,
        encoder=encoder.spec.key, encoder_revision=encoder.spec.revision,
        embed_dim=encoder.spec.dim, pixel_gate_tau=0.0, segment_tau=segment_tau,
        decode_encode_s=wall, frames_per_s=n / wall if wall else 0.0,
        realtime_factor=covered / wall if wall else 0.0, seed=seed)
    meta = {"format": INDEX_FORMAT, "stats": asdict(stats),
            "store_bytes": size_bytes,
            "video_bytes": os.path.getsize(video),
            "jpeg_quality": jpeg_quality, "store_max_pixels": store_max_pixels,
            "encoder_s": t_encode, "jpeg_s": t_jpeg, "lance_write_s": t_write}
    with open(os.path.join(out, "framesieve.json"), "w") as f:
        json.dump(meta, f, indent=2)

    if verbose:
        hrs = covered / 3600
        print(f"{n:,} frames @ {target_fps} fps -> {stats.n_segments:,} segments")
        print(f"  {wall:.1f} s wall ({stats.realtime_factor:.0f}x realtime): "
              f"encode {t_encode:.1f}s, jpeg {t_jpeg:.1f}s, "
              f"decode+write {t_write:.1f}s")
        print(f"  {size_bytes/1e9:.3f} GB store "
              f"({size_bytes/1e6/max(hrs,1e-9):.0f} MB per hour of video, "
              f"{size_bytes/max(1,os.path.getsize(video)):.2f}x the source video)")
    return meta


def _resize(arr: np.ndarray, size: int):
    from PIL import Image
    return Image.fromarray(arr).resize((size, size), Image.BILINEAR)


# --------------------------------------------------------------------------


class FrameStore:
    """Read side: vectors for selection, blobs for refinement, in one dataset."""

    def __init__(self, path: str, decode_workers: int = 8):
        lance = _require_lance()
        self.path = path
        meta_path = os.path.join(path, "framesieve.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"{path} is a Lance dataset but has no framesieve.json, so it "
                f"is not a complete framesieve store (an interrupted build "
                f"leaves this state). Delete the directory and rebuild.")
        self.ds = lance.dataset(path)
        self.decode_workers = decode_workers
        with open(meta_path) as f:
            self.meta = json.load(f)
        self.stats = IndexStats.from_dict(self.meta["stats"])
        tbl = self.ds.to_table(columns=["frame_idx", "ts", "seg_id"])
        self.frame_idx = tbl.column("frame_idx").to_numpy()
        self.ts = tbl.column("ts").to_numpy().astype(np.float32)
        self.seg_id = tbl.column("seg_id").to_numpy().astype(np.int32)
        self._emb: np.ndarray | None = None

    @property
    def emb(self) -> np.ndarray:
        """Embeddings, loaded lazily -- selection needs them, plain fetching does not.

        float32, same as the plain index: these embeddings sit in a band where
        neighbours differ in the third decimal, and a float16 round-trip here
        would quietly undo the 0.2.0 one-format decision for store-backed
        indexes only.
        """
        if self._emb is None:
            col = self.ds.to_table(columns=["emb"]).column("emb")
            self._emb = np.stack(col.to_numpy(zero_copy_only=False)).astype(np.float32)
        return self._emb

    def to_frame_index(self) -> FrameIndex:
        """A plain in-memory index, for the evaluation code paths."""
        return FrameIndex(self.ts, self.emb, self.seg_id, self.stats)

    def fetch(self, timestamps: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        """Frames at the given timestamps, by byte-range read rather than seek."""
        ts = np.asarray(list(timestamps), dtype=np.float64)
        if len(ts) == 0:
            return np.zeros(0, np.float32), np.zeros((0, 1, 1, 3), np.uint8)
        pos = np.searchsorted(self.ts, ts)
        pos = np.clip(pos, 0, len(self.ts) - 1)
        left = np.clip(pos - 1, 0, len(self.ts) - 1)
        pos = np.where(np.abs(self.ts[left] - ts) < np.abs(self.ts[pos] - ts),
                       left, pos)
        # take_blobs wants ascending row ids; restore the caller's order after
        order = np.argsort(pos)
        rows = [int(i) for i in pos[order]]
        blobs = self.ds.take_blobs("jpeg", indices=rows)
        raw = [b.read() for b in blobs]
        with ThreadPoolExecutor(self.decode_workers) as ex:
            arrs = list(ex.map(_decode_jpeg, raw))
        inv = np.argsort(order)
        return self.ts[pos], np.stack([arrs[i] for i in inv])

    def segments(self):
        bounds = np.flatnonzero(np.diff(self.seg_id)) + 1
        starts = np.concatenate([[0], bounds])
        ends = np.concatenate([bounds, [len(self.seg_id)]])
        return starts, ends, self.ts[starts], self.ts[ends - 1]

    def __len__(self) -> int:
        return len(self.ts)
