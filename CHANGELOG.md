# Changelog

All notable changes to this project are recorded here. This project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html); until 1.0 the public
API in `framesieve/api.py` may change, and anything under `framesieve.index`,
`.search`, `.encoders` or `.vlm` may change without notice.

## [0.1.0] — unreleased

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

### Searching many videos at once
`framesieve.Collection`, backed by LanceDB: frame vectors on disk with a vector
index over them, searched across a whole corpus rather than one video, returning
which video as well as when. Measured on 10M vectors (2,778 video-hours, 62 GB):
112 ms per query at 5.5 GB peak resident, against 31 GB to hold the same vectors
in memory. Runs under an 8 GB cap; OOM-killed at 4 GB.

Defaults to `IvfHnswFlat`. The quantized index types are unusable on these
embeddings — IvfPq scores 0% recall@20 and IvfRq 24%, because the similarities
being ranked span a band narrower than the quantization error.

### Runs without a GPU
CUDA if there is one, Apple silicon if there is one, CPU otherwise. Indexing an
hour of video takes about 1 minute on 64 CPU cores and 2 minutes on 8 threads,
against 15 seconds on a GH200; search is ~25 ms either way. Only the VLM confirm
stage really wants a GPU.

### Known limitations
- One GPU, h.264 input, English queries.
- The retrieval encoder is caption-trained: phrase queries as captions
  ("a dark tunnel"), not questions.
- No audio, no speech, no OCR. Text on signs is close to invisible to the
  retrieval stage (R@1 3.4 on MomentSeeker's OCR split).
