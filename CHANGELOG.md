# Changelog

All notable changes to this project are recorded here. This project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html); until 1.0 the public
API in `framesieve/api.py` may change, and anything under `framesieve.indexing`,
`.search`, `.encoders` or `.vlm` may change without notice.

## [Unreleased]

### Breaking
- **`framesieve search` is retrieval-only by default**, exactly like the Python
  API's `confirm=False`: a bare search returns ranked candidates in
  milliseconds and never downloads a model. `--confirm` opts in to VLM
  verification, which fetches the surviving frames and pulls ~16 GB of weights
  on first use (needs `framesieve[vlm]`).
- **The module `framesieve.index` is renamed `framesieve.indexing`.** The old
  name collided with the `index()` function, so `import framesieve.index as m`
  handed back the function rather than the module. Lower-level imports are now
  `from framesieve.indexing import FrameIndex, build_index`.
- **`index(segment_tau=)` defaults to 0.90 rather than 0.0**, matching the CLI
  and the store, so all three build paths collapse redundancy the same way.
  Pass `segment_tau=0` for the old no-collapse behaviour.

### Fixed
- `load()` encodes queries with the encoder the sidecar records, not whatever
  `encoder=` was passed; mixing the two produces plausible nonsense.
- `save()` on a store-backed index refuses instead of overwriting the stored
  frames with the embeddings-only table; a different explicit path exports the
  embeddings alone.
- An audio track with no speech, or a video with no legible on-screen text,
  saves an empty sidecar instead of crashing the pass.
- The CLI no longer crashes when the top hit came from the transcript rather
  than a frame.
- `framesieve index --out PATH` is honoured; an index at a custom path is
  searched by passing that path.
- `Collection.add_indexes` skips speech/OCR sidecars and already-added videos
  with a note instead of failing mid-load, raises on an encoder mismatch, and
  quotes filenames, so an apostrophe in a video name no longer breaks the
  filter.
- `--store` streams frames to disk as they are encoded instead of buffering the
  whole video in memory, and reads embeddings back as the float32 they are.
- `gpu_decode` follows the codec ffprobe reports instead of assuming h.264.
- `open()` on an existing index runs a missing `audio=`/`ocr=` pass instead of
  silently ignoring it, warns naming any other build option it cannot apply,
  and raises `TypeError` on a misspelled keyword.
- A fetch that fails for every requested frame raises instead of quietly
  returning nothing.
- The OCR pass preserves aspect ratio instead of distorting frames.
- `--json` keeps stdout machine-readable, with the commentary on stderr, and
  its hits now carry `source` and `text`.
- The index sidecar records a format version; unknown stats fields written by
  newer versions are tolerated, and a Lance directory with no `framesieve.json`
  gets a delete-and-rebuild error rather than a traceback.

## [0.2.0] — 2026-08-16

Everything below was added or fixed after 0.1.0 went to PyPI. **0.1.0 is broken
on machines without a GPU and its `--store` wrote an index nothing could read;
use this instead.**

### Fixed
- **Ran only on a GPU.** `index()` died on a CPU-only machine with
  `RuntimeError: No CUDA GPUs are available`, raised from inside torch. Now picks
  CUDA, then Apple silicon, then CPU, with `float32` on CPU because `bfloat16` is
  emulated there. Indexing an hour of video takes 1–2 minutes on a CPU against
  15 seconds on a GPU; search is ~110 ms against ~6 ms.
- **`--store` wrote an index that could not be read back.** It produced a
  `.lance` frame store that `search` never looked for.
- **A frameless index was mistaken for a frame store**, failing at the moment
  someone asked for a frame rather than when the index was opened.

### Searching many videos at once
`framesieve.Collection`, backed by LanceDB: frame vectors on disk with a vector
index over them, searched across a whole corpus rather than one video, returning
which video as well as when. Measured on 10M vectors (2,778 video-hours, 62 GB):
112 ms per query at 5.5 GB peak resident, against 31 GB to hold the same vectors
in memory. Runs under an 8 GB cap; OOM-killed at 4 GB.

Defaults to `IvfHnswFlat` at `nprobes=50`, which returns the same top hit as an
exact scan on 15 of 15 test queries in 10 ms. `IvfFlat` needs `nprobes=400` and
81 ms to match that. The quantized index types are unusable on these embeddings
— `IvfPq` scores 0% recall@20 and `IvfRq` 24%, because the similarities being
ranked span a band narrower than the quantization error.

None of this changes any existing number: single-video search is still a numpy
matrix multiply and never touches LanceDB, and the benchmark harnesses read the
same `.npz` sidecars they always did.

### On-screen text search
`framesieve index --ocr` reads the text in the frames and indexes it, so
`source="text"` reaches a caption, a slide title or a scoreboard — which the
retrieval stage cannot: it scores R@1 3.4 on MomentSeeker's OCR split, close to
chance.

Roughly 120 ms a frame, so `--ocr-every segment` (the default) reads one frame
per shot instead of every frame, reusing the redundancy the index already found:
2,255 frames became 547 reads on the test video, about 1.5 min per video-hour.
`--ocr-every frame` reads all of them for footage whose text changes under a
still picture.

Speech and OCR share one container, `framesieve.timedtext` — they arrive by
different routes and are the same thing once they arrive — and `search` merges
any number of sources rather than two.

### Speech search
`framesieve index --audio` / `index(video, audio=True)` transcribes with Whisper
and indexes the timed segments beside the frames, at about 11x realtime. Search
takes `source="visual"`, `"speech"`, or the default, which uses whatever the
index has.

The two are never ranked against each other: a frame similarity and a sentence
similarity are different quantities on different scales. Each modality is ranked
within itself and the two are merged on time, so a frame hit and a transcript hit
on the same moment return once, their sources joined by `+`
(`source="speech+visual"`), and are promoted — agreement between independent
signals beats either list's leader. Every `Hit` carries `.source` and, for
speech, `.text`.

Frames use SigLIP and speech uses a sentence encoder, because SigLIP's text tower
is built to sit beside images and is a poor text-to-text matcher.

### One index format
Indexes are Lance datasets, not compressed npz. Same container as the frame
store and `Collection`, so there is one format rather than three; opens 4x
faster (35 ms against 145 ms for a 4.5-hour video); costs 11 MB per hour of
video against 5. Embeddings are stored float32, which removes the widening cast
entirely.

`pylance` moves from an optional extra to a core dependency; `framesieve[store]`
stays, and now maps to pillow, which the frame store's JPEG encoding needs.

**Breaking:** `.npz` indexes no longer load. `FrameIndex.from_npz` reads the old
form and `scripts/convert_indexes.py` migrates a directory of them; neither is
on the library's path. `FrameIndex.emb32` is gone, since `emb` is float32.

### Search is 4x faster
A float16 to float32 cast was running on every query rather than once per index:
17.13 ms of a 17.42 ms search on a 4.5-hour video, against 0.03 ms for the
matrix multiply it fed. Cached on `FrameIndex.emb32`, so a search over 16,244
frames goes from 24.6 ms to 6.3 ms, of which 5.2 ms is now encoding the query
text. Results are unchanged.

The cache costs 4 bytes per dimension per frame — 50 MB for 4.5 hours, 1.1 GB
for a hundred — which is the same arithmetic that decides when to move to a
`Collection`.

### Runs without a GPU
CUDA if there is one, Apple silicon if there is one, CPU otherwise. Indexing an
hour of video takes about 1 minute on 64 CPU cores and 2 minutes on 8 threads,
against 15 seconds on a GH200; search is ~110 ms on CPU against ~6 ms on a GPU.
Only the VLM confirm stage really wants a GPU.

## [0.1.0] — 2026-08-16

First public release.

### Added
- `framesieve.open` / `.index` / `.load`, returning a `VideoIndex` you can
  `search`, `score` and pull `frames` from.
- `framesieve` command with `index`, `search` and `info`, all supporting
  `--json` for use in a pipeline.
- Five candidate-selection strategies behind one interface (`uniform`, `topk`,
  `nms`, `segment`, `segment_adaptive`).
- Optional Lance frame store (`--store` / `index(store=True)`): every sampled
  frame kept as a JPEG beside its embedding. Fetching a frame goes from 14.5 ms
  to 0.9 ms and the index stops needing the source video, at 275 MB per hour of
  disk against 5 MB and about half the indexing throughput. Off by default.
- `framesieve.pooling`: `topk_mean`, `recommend_k`, `k_range` and `sweep_k`,
  numpy-only and independent of the rest of the package.

### Measured
- Indexing runs at 105× realtime and costs about 15 s and 5 MB per hour of video.
- On MomentSeeker, R@1 20.10 from retrieval alone and 23.40 with ten VLM calls
  per query, against InternVideo2's published 19.70.
- Against dense-VLM ground truth on a 4.5 h video, 26× the event recall of
  uniform sampling at 32 model calls.

### Known limitations
- One GPU, h.264 input, English queries.
- The retrieval encoder is caption-trained: phrase queries as captions
  ("a dark tunnel"), not questions.
- No audio, no speech, no OCR. Text on signs is close to invisible to the
  retrieval stage (R@1 3.4 on MomentSeeker's OCR split).
