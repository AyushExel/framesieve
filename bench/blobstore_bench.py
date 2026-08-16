"""Is a Lance blob store a better frame source than seeking the video?

The refine stage needs a few dozen arbitrary frames at full resolution. Today it
seeks the video with ffmpeg, which costs 121 ms/frame single-threaded and 14 ms
across 32 workers -- against a 107 ms VLM call, that is not free.

The alternative: while indexing, we have already decoded every sampled frame. We
can JPEG-encode it and write it into a Lance dataset as a blob column, alongside
its embedding. Then "fetch the frame at t" becomes a point lookup and a byte-range
read instead of a seek-and-decode.

The tradeoff is storage, so this measures both: latency *and* bytes. If the store
costs more disk than the video it came from, that needs saying out loud.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import time

import numpy as np
import pyarrow as pa

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from framesieve.fetch import FrameFetcher  # noqa: E402
from framesieve.frames import FrameStream, probe_source  # noqa: E402

BLOB_META = {"lance-encoding:blob": "true"}


def jpeg_bytes(arr: np.ndarray, quality: int) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality,
                              optimize=False, progressive=False)
    return buf.getvalue()


def build_store(video: str, out: str, *, fps: float, quality: int,
                max_frames: int = 0) -> dict:
    import lance

    if os.path.exists(out):
        shutil.rmtree(out)
    info = probe_source(video)
    stream = FrameStream(video, target_fps=fps, size=None, batch=64)

    schema = pa.schema([
        pa.field("ts", pa.float64()),
        pa.field("frame_idx", pa.int32()),
        pa.field("jpeg", pa.large_binary(), metadata=BLOB_META),
    ])

    t_decode = t_encode = 0.0
    n = 0
    batches = []
    t_last = time.perf_counter()
    for ts, frames in stream:
        t_decode += time.perf_counter() - t_last
        t0 = time.perf_counter()
        jp = [jpeg_bytes(f, quality) for f in frames]
        t_encode += time.perf_counter() - t0
        batches.append(pa.record_batch(
            [pa.array(ts, pa.float64()),
             pa.array(np.arange(n, n + len(frames)), pa.int32()),
             pa.array(jp, pa.large_binary())], schema=schema))
        n += len(frames)
        if max_frames and n >= max_frames:
            break
        t_last = time.perf_counter()

    t0 = time.perf_counter()
    lance.write_dataset(pa.Table.from_batches(batches, schema=schema), out,
                        mode="create", data_storage_version="stable")
    t_write = time.perf_counter() - t0

    size = sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, fs in os.walk(out) for f in fs)
    return {"n_frames": n, "store_bytes": size, "video_bytes": os.path.getsize(video),
            "decode_s": t_decode, "jpeg_encode_s": t_encode, "lance_write_s": t_write,
            "video_duration_s": info.duration_s, "quality": quality,
            "resolution": f"{info.width}x{info.height}"}


def bench_lance(store: str, ts_all: np.ndarray, picks: np.ndarray,
                repeats: int) -> dict:
    import lance
    from PIL import Image

    ds = lance.dataset(store)
    idx = [int(i) for i in picks]

    # warm
    _ = ds.take_blobs("jpeg", indices=idx[:4])

    lat = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        blobs = ds.take_blobs("jpeg", indices=idx)
        raw = [b.read() for b in blobs]
        t_read = time.perf_counter() - t0
        t1 = time.perf_counter()
        arrs = [np.array(Image.open(io.BytesIO(r)).convert("RGB")) for r in raw]
        t_dec = time.perf_counter() - t1
        lat.append((t_read, t_dec))
        assert len(arrs) == len(idx)
    reads = np.array([x[0] for x in lat])
    decs = np.array([x[1] for x in lat])
    return {"n": len(idx),
            "read_s_median": float(np.median(reads)),
            "jpeg_decode_s_median": float(np.median(decs)),
            "total_s_median": float(np.median(reads + decs)),
            "ms_per_frame": float(np.median(reads + decs)) / len(idx) * 1000,
            "ms_per_frame_read_only": float(np.median(reads)) / len(idx) * 1000}


def bench_ffmpeg(video: str, ts: np.ndarray, workers: int, repeats: int) -> dict:
    f = FrameFetcher(video, workers=workers)
    lat = []
    f.fetch(ts[:4].tolist())          # warm the page cache
    for _ in range(repeats):
        t0 = time.perf_counter()
        got, frames = f.fetch(ts.tolist())
        lat.append(time.perf_counter() - t0)
        assert len(frames) == len(ts), f"{len(frames)} != {len(ts)}"
    m = float(np.median(lat))
    return {"workers": workers, "total_s_median": m,
            "ms_per_frame": m / len(ts) * 1000}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/demo_clip.mp4")
    ap.add_argument("--store", default="runs/blobstore_demo.lance")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--n-fetch", type=int, nargs="*", default=[8, 32, 128])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--workers", type=int, nargs="*", default=[1, 8, 32])
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--out", default="runs/blobstore_bench.json")
    args = ap.parse_args()

    if args.rebuild or not os.path.exists(args.store):
        print(f"building blob store from {args.video} ...")
        binfo = build_store(args.video, args.store, fps=args.fps, quality=args.quality)
        with open(args.store + ".meta.json", "w") as f:
            json.dump(binfo, f, indent=2)
    else:
        binfo = json.load(open(args.store + ".meta.json"))

    hrs = binfo["video_duration_s"] / 3600
    print(f"\nstore: {binfo['n_frames']:,} frames @ {binfo['resolution']} "
          f"jpeg q{binfo['quality']}")
    print(f"  size      {binfo['store_bytes']/1e9:.3f} GB "
          f"({binfo['store_bytes']/1e6/max(hrs,1e-9):.0f} MB per hour of video)")
    print(f"  source    {binfo['video_bytes']/1e9:.3f} GB "
          f"-> store is {binfo['store_bytes']/binfo['video_bytes']:.2f}x the video")
    print(f"  build     decode {binfo['decode_s']:.1f}s + jpeg "
          f"{binfo['jpeg_encode_s']:.1f}s + write {binfo['lance_write_s']:.1f}s")

    import lance
    ds = lance.dataset(args.store)
    ts_all = ds.to_table(columns=["ts"]).column("ts").to_numpy()
    rng = np.random.default_rng(0)

    rows = []
    print(f"\n{'n frames':>9}  {'source':<22}{'total ms':>10}{'ms/frame':>10}")
    print("-" * 54)
    for n in args.n_fetch:
        if n > len(ts_all):
            continue
        picks = rng.choice(len(ts_all), size=n, replace=False)
        picks.sort()
        r = bench_lance(args.store, ts_all, picks, args.repeats)
        rows.append({"n": n, "source": "lance blob", **r})
        print(f"{n:>9}  {'lance blob':<22}{r['total_s_median']*1000:>10.1f}"
              f"{r['ms_per_frame']:>10.2f}")
        print(f"{'':>9}  {'  (byte-range read)':<22}"
              f"{r['read_s_median']*1000:>10.1f}{r['ms_per_frame_read_only']:>10.2f}")
        for w in args.workers:
            rf = bench_ffmpeg(args.video, ts_all[picks], w, args.repeats)
            rows.append({"n": n, "source": f"ffmpeg seek x{w}", **rf})
            print(f"{n:>9}  {'ffmpeg seek x'+str(w):<22}"
                  f"{rf['total_s_median']*1000:>10.1f}{rf['ms_per_frame']:>10.2f}")
        print()

    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "store": binfo, "rows": rows}, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
