# Getting frames back out

The refine stage needs a few dozen arbitrary frames at full resolution. That
sounds trivial and is not: done naively it costs more than the VLM call it feeds.

Measured on `data/demo_clip.mp4` (400 s, 960×720, h264, 2.1 s GOP), GH200 host.

## Random single-frame access

This is the access pattern the cascade actually has: 8–128 scattered frames.

| n | method | total ms | ms/frame |
|---:|---|---:|---:|
| 32 | **lance jpeg blob** | 31.5 | **0.98** |
| 32 | ffmpeg seek ×32 workers | 410.1 | 12.82 |
| 32 | ffmpeg seek ×1 worker | 3440.5 | 107.52 |
| 32 | lance video byte-range | 2851.0 | 89.09 |
| 128 | **lance jpeg blob** | 131.1 | **1.02** |
| 128 | ffmpeg seek ×32 workers | 1805.1 | 14.10 |
| 128 | lance video byte-range | 9616.7 | 75.13 |

The JPEG blob store wins by 13× over the best ffmpeg path and by 105× over the
naive one. Of its ~1 ms, the byte-range read is **0.09 ms** — the rest is JPEG
decode, which parallelises across 8 threads and would go to near zero on nvJPEG.

## Why the video byte-range approach loses here

Storing the video once and reading only the byte ranges you need is the more
elegant design — no duplication, lossless, and it works over object storage where
seeking is not an option. It is also, for *this* access pattern, much slower, and
the reason is structural rather than an implementation detail:

**To decode one frame you must decode its whole GOP.** At a 2.1 s GOP and 25 fps
that is ~60 frames of work to produce one — a 60× decode amplification. Even with
an in-process decoder the arithmetic does not rescue it:

| step | cost |
|---|---|
| ffmpeg subprocess per GOP (60 frames) | 224.7 ms |
| PyAV in-process per GOP (60 frames) | 183.7 ms |
| PyAV, stop at the first frame | 19.5 ms |

So the floor for an arbitrary frame is roughly "open the container, decode to the
target" ≈ 20 ms even in-process — still 20× the JPEG store. Process spawn was not
the problem; the decode amplification is.

## Where the video byte-range approach does win

Three places, and they are real:

1. **Object storage.** `fetch_blob_ranges` coalesces and schedules the reads
   ([lancedb#3703](https://github.com/lancedb/lancedb/pull/3703)). Over S3 there
   is no cheap local seek to compare against.
2. **Clip-shaped access.** If you want *every* frame in a window, the GOP decode
   is not wasted:

   | window | method | frames | ms/frame |
   |---|---|---:|---:|
   | 8 s | lance video byte-range | 201 | 2.25 |
   | 8 s | ffmpeg decode window | 200 | 1.83 |

   Comparable locally, and the only option remotely.
3. **It has every frame.** The JPEG store only holds what was sampled — at 1 fps,
   one frame per second. The video blob still contains all 25 fps, so it can
   answer "give me the half second around this hit" at full temporal detail.
   Nothing in the JPEG store can.

## What framesieve does

Default: **JPEG blob store** (`./framesieve index VIDEO --store`), because the
cascade's access pattern is scattered single frames and that is what it is best
at. It costs 275 MB per hour of video, which is **0.30× the source file** — one
JPEG per second is smaller than 25 H.264 frames per second, so this is cheaper
than it sounds.

Fallback with no `pylance` installed: parallel ffmpeg seeks. Correct, 13× slower,
no extra disk.

`bench/videoblob.py` implements the video byte-range store too, verified
frame-exact against a plain seek (mean abs diff **0.000**, 8/8 identical). It is
the right tool for remote storage and for clip-shaped reads; it is not the right
tool for the refine stage on a local disk, and the numbers above are why.

## Correctness notes

- The JPEG store is lossy. At q90 it changed **zero** VLM decisions across 400
  frame-query pairs; scores move by ~0.3 on a scale whose std is 3.2.
- The video byte-range store is lossless: frames come out of the original
  bitstream and match a plain seek exactly.
- PyAV and ffmpeg disagree slightly on YUV→RGB conversion (mean abs diff 1.15),
  so the two decoders are not interchangeable at the byte level. framesieve uses
  ffmpeg throughout.
- MPEG-TS remuxing restamps time: ffmpeg's TS muxer starts at 1.4 s and MP4s
  often carry a negative first PTS. The offset is **measured** at build time
  (1.60 s for this clip) and verified against a real seek, never assumed.
