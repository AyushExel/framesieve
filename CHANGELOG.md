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
- Optional Lance frame store (`--store`): frames kept as JPEG blobs beside their
  embeddings, making the refine stage's frame fetch about 14× faster for roughly
  0.3× the video in disk.
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
