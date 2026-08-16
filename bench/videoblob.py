"""The design that was measured and not adopted: store the video itself.

Kept as a benchmark rather than as library code. It remuxes to TS, indexes the
GOPs, and stores the whole video as one Lance blob, so a frame comes back as a
byte-range read over the container instead of an ffmpeg seek. It works, and
docs/frame-access.md has the numbers, but framesieve ships the per-frame JPEG
store instead: this one pays a GOP decode per frame, which is the cost the
frame store removes outright.

It also uses lancedb, which framesieve does not depend on -- another reason it
belongs here and not in the package.

Store the video once, read only the byte ranges you need.

The JPEG store in store.py is fast but duplicates pixels: it keeps a second copy
of every sampled frame. LanceDB's batched blob range reads
(lancedb/lancedb#3703) allow the other design -- keep the *video* as a single
blob, keep a small GOP index beside it, and read only the byte ranges that cover
the frames you want. That is:

  - no duplication: the video is stored as-is
  - lossless: frames come out of the original bitstream, not a JPEG round-trip
  - remote-friendly: Lance coalesces and schedules the reads, so this works over
    object storage where seeking a file would be brutal
  - clip-shaped: one range read yields a whole GOP, so asking for two seconds of
    video costs the same as asking for one frame of it

The container matters. An arbitrary byte slice of an MP4 is not decodable -- the
headers live in the moov box at the other end of the file. So the video is
remuxed (stream copy, no re-encode) to MPEG-TS, which is self-describing: every
range starting at a keyframe decodes on its own and carries its own timestamps.

The one subtlety is time. ffmpeg's TS muxer restamps to a 1.4 s origin, and MP4s
frequently carry a negative first PTS from encoder delay, so TS time and source
time differ by a constant that has to be measured rather than assumed. It is
computed at build time and verified against a real seek.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# bench/ is not part of the package, so reach the library the same way the
# other benchmarks here do
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))
from framesieve.frames import probe_source  # noqa: E402

_PTS_RE = re.compile(rb"pts_time:\s*([0-9.\-]+)")


def _ffprobe_packets(path: str, fmt: str | None = None) -> list[tuple[float, int, str]]:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0"]
    if fmt:
        cmd += ["-f", fmt]
    cmd += ["-show_entries", "packet=pts_time,pos,flags", "-of", "csv=p=0", path]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    rows = []
    for line in out.splitlines():
        p = line.strip().split(",")
        if len(p) < 3:
            continue
        try:
            rows.append((float(p[0]), int(p[1]), p[2]))
        except ValueError:
            continue
    return rows


@dataclass
class GopEntry:
    idx: int
    byte_pos: int
    byte_len: int
    ts_pts: float          # timestamp inside the remuxed TS
    t_start: float         # source-video time of the first frame
    t_end: float           # source-video time of the last frame (inclusive)


def build_video_blob(video: str, out_uri: str, *, table_name: str = "video",
                     ts_path: str | None = None, keep_ts: bool = False,
                     verify: bool = True) -> dict:
    """Remux to TS, index its GOPs, and store the TS as a single Lance blob."""
    import lancedb
    import pyarrow as pa

    info = probe_source(video)
    t0 = time.perf_counter()

    ts_path = ts_path or (os.path.splitext(video)[0] + ".framesieve.ts")
    if not os.path.exists(ts_path):
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-v", "error", "-y", "-i", video,
             "-c:v", "copy", "-an", "-sn", "-f", "mpegts", ts_path], check=True)
    t_remux = time.perf_counter() - t0

    src_pkts = _ffprobe_packets(video)
    ts_pkts = _ffprobe_packets(ts_path)
    if not src_pkts or not ts_pkts:
        raise RuntimeError("could not read packet tables")

    # TS time - source packet time. ffmpeg's TS muxer starts at 1.4 s; MP4s often
    # start negative from encoder delay. Neither is safe to assume.
    offset = ts_pkts[0][0] - src_pkts[0][0]
    # source *display* time starts at 0 even when the first packet pts is negative
    display_shift = -src_pkts[0][0]

    keys = [(i, p) for i, (pts, p, fl) in enumerate(ts_pkts) if "K" in fl]
    size = os.path.getsize(ts_path)
    gops: list[GopEntry] = []
    for k, (pi, pos) in enumerate(keys):
        end_pos = keys[k + 1][1] if k + 1 < len(keys) else size
        pts = ts_pkts[pi][0]
        last_pi = (keys[k + 1][0] - 1) if k + 1 < len(keys) else (len(ts_pkts) - 1)
        # `-ss T` addresses the packet timeline, which is what the rest of
        # framesieve uses, so the display shift is deliberately NOT applied here
        gops.append(GopEntry(
            idx=k, byte_pos=pos, byte_len=end_pos - pos, ts_pts=pts,
            t_start=pts - offset, t_end=ts_pkts[last_pi][0] - offset))
    gops.sort(key=lambda g: g.t_start)

    # blob *v2* specifically: fetch_blob_ranges rejects the legacy
    # `lance-encoding:blob` column, and v2 is what carries range-read support
    from lancedb.schema import BlobType

    db = lancedb.connect(out_uri)
    with open(ts_path, "rb") as f:
        blob = f.read()
    bt = BlobType()
    storage = pa.array([{"data": blob, "uri": None, "position": None, "size": None}],
                       type=bt.storage_type)
    arr = pa.ExtensionArray.from_storage(bt, storage)
    if table_name in db.table_names():
        db.drop_table(table_name)
    db.create_table(table_name, pa.table(
        {"video_id": pa.array([os.path.basename(video)]), "bytes": arr}))

    meta = {
        "video": os.path.abspath(video), "ts_path": os.path.abspath(ts_path),
        "duration_s": info.duration_s, "fps": info.fps,
        "width": info.width, "height": info.height,
        "n_gops": len(gops), "ts_bytes": size, "video_bytes": os.path.getsize(video),
        "ts_offset": offset, "display_shift": display_shift,
        "mean_gop_s": float(np.mean([g.t_end - g.t_start for g in gops])),
        "remux_s": t_remux, "build_s": time.perf_counter() - t0,
        "gops": [g.__dict__ for g in gops],
    }
    with open(os.path.join(out_uri, f"{table_name}.framesieve.json"), "w") as f:
        json.dump(meta, f)

    if verify:
        store = VideoBlobStore(out_uri, table_name)
        meta["verify"] = store.verify_alignment(video)
    if not keep_ts:
        os.remove(ts_path)
    return meta


class VideoBlobStore:
    """Random frame and clip access by byte range over a single stored video."""

    def __init__(self, uri: str, table_name: str = "video", decode_workers: int = 16):
        import lancedb

        self.uri = uri
        self.table_name = table_name
        self.decode_workers = decode_workers
        self.db = lancedb.connect(uri)
        self.tbl = self.db.open_table(table_name)
        with open(os.path.join(uri, f"{table_name}.framesieve.json")) as f:
            self.meta = json.load(f)
        self.gops = [GopEntry(**g) for g in self.meta["gops"]]
        self.g_start = np.array([g.t_start for g in self.gops])
        self.g_end = np.array([g.t_end for g in self.gops])
        rows = self.tbl.search().with_row_id(True).limit(1).to_arrow().to_pylist()
        self.row_id = int(rows[0]["_rowid"])

    # -- lookup ------------------------------------------------------------

    def gop_for(self, t: float) -> int:
        i = int(np.searchsorted(self.g_start, t, "right") - 1)
        return int(np.clip(i, 0, len(self.gops) - 1))

    def gops_for_range(self, t0: float, t1: float) -> list[int]:
        a, b = self.gop_for(t0), self.gop_for(t1)
        return list(range(a, b + 1))

    # -- decode ------------------------------------------------------------

    def _decode(self, raw: bytes, t_base: float) -> tuple[np.ndarray, np.ndarray]:
        """Decode a TS byte range; return (source_times, frames).

        ffmpeg re-bases timestamps to zero when it is handed a fragment, so the
        absolute time comes from the GOP index and only the *relative* spacing
        comes from the decoded stream.
        """
        W, H = self.meta["width"], self.meta["height"]
        p = subprocess.run(
            ["ffmpeg", "-hide_banner", "-v", "info", "-nostats", "-nostdin",
             "-f", "mpegts", "-i", "pipe:0", "-an", "-sn", "-vsync", "0",
             "-vf", "showinfo", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"],
            input=raw, capture_output=True)
        fb = W * H * 3
        n = len(p.stdout) // fb
        if n == 0:
            return np.zeros(0), np.zeros((0, H, W, 3), np.uint8)
        frames = np.frombuffer(p.stdout[: n * fb], np.uint8).reshape(n, H, W, 3)
        pts = [float(m) for m in _PTS_RE.findall(p.stderr)][:n]
        while len(pts) < n:
            pts.append((pts[-1] if pts else 0.0) + 1.0 / max(self.meta["fps"], 1e-6))
        rel = np.array(pts) - pts[0]
        return t_base + rel, frames

    def _decode_many(self, gop_ids: Sequence[int], chunks
                     ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Decode several GOPs at once.

        Each GOP is an independent ffmpeg process, so this is embarrassingly
        parallel -- and it has to be, because one range read yields a whole GOP
        (~60 frames) whether you wanted one frame from it or all of them.
        """
        from concurrent.futures import ThreadPoolExecutor

        raws = [(c.as_py() if hasattr(c, "as_py") else bytes(c)) for c in chunks]
        bases = [self.gops[g].t_start for g in gop_ids]
        with ThreadPoolExecutor(min(self.decode_workers, max(1, len(raws)))) as ex:
            outs = list(ex.map(lambda a: self._decode(a[0], a[1]), zip(raws, bases)))
        pool_t = [t for t, f in outs if len(t)]
        pool_f = [f for t, f in outs if len(t)]
        return pool_t, pool_f

    def fetch(self, timestamps: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        """Frames nearest the given source times, via batched blob range reads."""
        ts = np.asarray(list(timestamps), dtype=np.float64)
        if len(ts) == 0:
            return np.zeros(0), np.zeros((0, 1, 1, 3), np.uint8)
        want_gops = sorted({self.gop_for(float(t)) for t in ts})
        reqs = [(self.row_id, self.gops[g].byte_pos, self.gops[g].byte_len)
                for g in want_gops]
        chunks = self.tbl.fetch_blob_ranges("bytes", reqs)

        pool_t, pool_f = self._decode_many(want_gops, chunks)
        if not pool_t:
            return np.zeros(0), np.zeros((0, 1, 1, 3), np.uint8)
        allt = np.concatenate(pool_t)
        allf = np.concatenate(pool_f)
        pick = np.array([int(np.argmin(np.abs(allt - t))) for t in ts])
        return allt[pick], allf[pick]

    def fetch_clip(self, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
        """Every frame between t0 and t1, from one batched range read."""
        gs = self.gops_for_range(t0, t1)
        reqs = [(self.row_id, self.gops[g].byte_pos, self.gops[g].byte_len) for g in gs]
        chunks = self.tbl.fetch_blob_ranges("bytes", reqs)
        pool_t, pool_f = self._decode_many(gs, chunks)
        if not pool_t:
            return np.zeros(0), np.zeros((0, 1, 1, 3), np.uint8)
        allt = np.concatenate(pool_t); allf = np.concatenate(pool_f)
        m = (allt >= t0 - 1e-6) & (allt <= t1 + 1e-6)
        return allt[m], allf[m]

    # -- checks ------------------------------------------------------------

    def verify_alignment(self, video: str, n: int = 8, seed: int = 0) -> dict:
        """A byte-range frame must equal the frame a plain seek returns.

        The TS/source time offset is measured, not assumed, so it gets checked.
        """
        from framesieve.fetch import FrameFetcher

        rng = np.random.default_rng(seed)
        ts = rng.uniform(2.0, max(3.0, self.meta["duration_s"] - 2.0), size=n)
        got_t, got_f = self.fetch(ts.tolist())
        ref_t, ref_f = FrameFetcher(video, workers=8).fetch(got_t.tolist())
        k = min(len(got_f), len(ref_f))
        if k == 0:
            return {"ok": False, "reason": "no frames"}
        d = np.array([np.abs(a.astype(np.int16) - b.astype(np.int16)).mean()
                      for a, b in zip(got_f[:k], ref_f[:k])])
        return {"ok": bool((d < 0.5).all()), "n": int(k),
                "mean_abs_diff": float(d.mean()), "max_abs_diff": float(d.max()),
                "identical_fraction": float((d < 0.5).mean())}
