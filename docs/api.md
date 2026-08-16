# API reference

Everything here is `framesieve.api`, re-exported at the top level. That module is
the stable surface. `framesieve.index`, `.search`, `.encoders`, `.vlm` and
`.frames` are the lower level, are useful, and may change between minor versions.

```python
import framesieve as fs
```

---

## Getting a `VideoIndex`

### `fs.open(video, *, encoder=..., fps=1.0, vlm=..., rebuild=False, **kwargs)`

Load this video's index, building it first if it does not exist. The call most
programs want.

```python
video = fs.open("holiday.mp4")
```

`rebuild=True` forces a fresh index even if a sidecar is there. Extra keyword
arguments are passed to `fs.index` when it has to build.

### `fs.index(video, *, encoder=..., fps=1.0, save=True, ...)`

Decode at `fps`, embed every frame, write a sidecar. Costs roughly **15 seconds
and 5 MB per hour** of video on one GPU. Needs torch.

| argument | default | what it does |
|---|---|---|
| `encoder` | `"siglip2-base-224"` | the per-frame encoder; see `framesieve.encoders.SIGLIP_MODELS` |
| `fps` | `1.0` | frames sampled per second of video |
| `save` | `True` | write the sidecar; `False` keeps it in memory only |
| `size` | `256` | decode resolution before encoding |
| `segment_tau` | `0.0` | cosine similarity below which a new segment starts; collapses static footage |
| `pixel_gate_tau` | `0.0` | skip encoding frames within this mean grey-level difference of the last kept one |
| `start`, `duration` | `0.0` | index only part of the video, in seconds |
| `gpu_decode` | `False` | decode with NVDEC: fewer CPU cores, slower wall clock on most hosts |
| `device` | auto | `"cuda"`, `"mps"` or `"cpu"`; picks whichever is present |
| `store` | `False` | also keep every frame as a JPEG beside its embedding: 15× faster frame fetch and no need for the video afterwards, at 55× the disk. Needs `pylance` |
| `seed` | `0` | |

### `fs.load(path_or_video, *, video=None, encoder=..., fps=1.0)`

Load an existing index, given either the sidecar path or the video it came from.
**Needs no GPU and no model.** Pass `video=` if the source file has moved and you
still want `confirm=True` or `frames()` to work.

### `fs.index_path_for(video, encoder=..., fps=1.0, store=False) -> str`

Where the sidecar for that combination lives. The encoder and rate are in the
filename because an index built with a different encoder is not interchangeable,
and silently reusing one would produce plausible nonsense.

---

## `VideoIndex`

| attribute | type | |
|---|---|---|
| `.video` | `str` | path to the source video |
| `.path` | `str \| None` | path to the sidecar, if it was saved or loaded |
| `.duration` | `float` | seconds |
| `.times` | `ndarray[float64]` | timestamp of every indexed frame |
| `.embeddings` | `ndarray[float32]` | one L2-normalised row per frame |
| `.stats` | `IndexStats` | encoder, revision, frame counts, throughput |
| `.frame_index` | `FrameIndex` | the underlying object, for the lower-level modules |
| `len(video)` | `int` | number of indexed frames |

### `.search(query, k=32, *, confirm=False, question=None, strategy="segment_adaptive", tokens_per_frame=64, seed=0) -> SearchResults`

Find the `k` moments most likely to match `query`.

- **`query`** — phrase it as a caption (`"a dark tunnel"`), not a question. The
  retrieval encoder is caption-trained and the difference is worth real accuracy.
- **`confirm`** — show the surviving frames to a vision-language model and return
  its verdict. About 30 ms per frame. Without it, a hit means "looks similar",
  which is not the same as "is".
- **`question`** — the yes/no question put to the model. Defaults to
  `"Does this frame show: {query}?"`.
- **`strategy`** — how candidates are spread over the video. See below.

Results are ordered by the model's verdict when there is one, by retrieval
similarity otherwise, so `results[0]` is always the best answer available.

### `.score(query) -> ndarray`

Retrieval similarity for **every** indexed frame, aligned with `.times`. Use it
when you want the whole curve rather than the top few — plotting where in a video
a concept appears, or finding every run above some level.

```python
curve = video.score("a dark tunnel")
import numpy as np
peaks = video.times[curve > np.percentile(curve, 99)]
```

`query` may also be a **precomputed embedding** of the same dimension, in which
case this needs numpy and nothing else — no torch, no model, no GPU:

```python
# on a GPU box
from framesieve.encoders import SiglipEncoder
qv = SiglipEncoder("siglip2-base-224").encode_text(["a dark tunnel"])[0].cpu().numpy()

# anywhere else
curve = framesieve.load("holiday.npz").score(qv)
```

That is the shape for serving: one machine holds the encoder, every other
machine holds indexes and ranks against them.

### `.frames(times, size=None) -> list[ndarray]`

Fetch the actual pixels at those timestamps, as `uint8` HWC arrays. Accepts a
`SearchResults` directly, so `video.frames(hits[:4])` works. Needs the source
video on disk.

### `.save(path=None) -> str`

Write the sidecar. Returns where it went.

---

## `SearchResults`

A sliceable, iterable sequence of `Hit` that also carries the timings.

| | |
|---|---|
| `len(r)`, `r[0]`, `r[:5]`, `for hit in r` | slicing returns a `SearchResults` |
| `.query` | the string you searched for |
| `.confirmed` | whether a VLM looked |
| `.times` | `ndarray` of hit times |
| `.timings` | `{"select_s", "fetch_s", "vlm_s"}` |
| `.latency_ms` | their sum |
| `.to_dicts()` | JSON-ready |
| `.above(threshold=0.0)` | only the hits the model scored above `threshold` |

`.above()` raises `ValueError` unless `confirm=True` was used. A retrieval
similarity has no absolute scale, so thresholding it would be a number that means
nothing — better to refuse than to filter plausibly.

## `Hit`

| | |
|---|---|
| `.time` | seconds from the start |
| `.timecode` | `"1:20:21"` |
| `.score` | retrieval similarity, comparable **within** a query only |
| `.vlm_score` | log-odds from the model, or `None` |
| `.confirmed` | `True` / `False` / `None` if no model has looked |

---

## Selection strategies

How `k` candidates are spread over a video. Real footage repeats itself, so the
naive answer returns `k` near-copies of one moment.

| strategy | what it does |
|---|---|
| `uniform` | evenly spaced in time, random phase. The baseline, and stronger than people expect |
| `topk` | the k highest-scoring frames. Simple, and prone to k near-copies |
| `nms` | top-k with temporal suppression, window adapted to the budget |
| `segment` | top-k over segments fixed at index time |
| `segment_adaptive` | segments cut per query, `8 × budget` of them. **Default**, and the best on our ground truth |

---

## `Collection` — many videos at once

`VideoIndex` is one video in memory. `Collection` is any number of them in
LanceDB, on disk. **[When and how to switch](scaling.md)** covers the decision;
this is the surface.

```python
lib = fs.Collection("library.lancedb")      # created if it does not exist
```

| | |
|---|---|
| `len(lib)` | frames in the collection |
| `.videos()` | the video paths in it |
| `.uri` | where it lives |

### `.add(video, **kwargs)` / `.add_index(path)` / `.add_indexes(glob)`

`add` indexes a video and appends it, and needs torch. `add_index` appends a
`.npz` sidecar that already exists and needs nothing — which is how a corpus is
loaded, since indexing is the expensive half and it is per-video.

```python
lib.add_indexes("footage/*.npz")     # merge what you already built
lib.add("new_camera.mp4")            # or index straight in
```

### `.build_ann(kind="hnsw", *, num_partitions=None, metric="cosine")`

Build the vector index. Do this **once, after the bulk load** — until you do,
every search scans the whole table.

`kind` is `"hnsw"` (default), `"hnsw_sq"`, `"flat"` or `"sq"`. The default gets
the same top hit as an exact scan on 15 of 15 test queries in 10 ms; `"flat"`
needs `nprobes=400` and 81 ms to match it. The quantized types are **unusable**
on these embeddings — see [scaling.md](scaling.md).

### `.search(query, k=20, *, video=None, exact=False, nprobes=50, min_gap_s=30.0, per_video=None) -> list[CollectionHit]`

The k best moments anywhere in the collection.

- **`video`** — restrict to one video; a filter, not a separate index
- **`exact`** — scan everything instead of using the index. Slower, and what to
  compare against
- **`min_gap_s`** — collapse hits closer together than this within a video. At
  1 fps consecutive frames are near-identical, so without it a top-5 is one
  moment five times. `0` disables
- **`per_video`** — at most this many hits from any one video

`CollectionHit` carries `.video`, `.time`, `.timecode` and `.score`.

### `.recall_at(queries, k=20, nprobes=50) -> float`

Run each query both exactly and approximately, and report the share of the exact
top-k the approximate search also found. Worth doing once per corpus: the right
`nprobes` is a property of how your vectors are distributed, not a constant.

---

## `framesieve.pooling`

Independent of everything above: numpy only, imports nothing else from this
package. It collapses many fine-grained scores into one, with the depth as an
explicit parameter instead of a `max` you wrote once.

```python
from framesieve.pooling import topk_mean, recommend_k, k_range, sweep_k

scores = topk_mean(per_subunit_scores, k=3)     # k=1 is max, k=n is mean
k      = recommend_k(sub_unit_is_relevant)      # from ~20 labelled examples
r      = sweep_k(cand_scores, is_positive, query_id, ks=k_range(labels))
r["best_k"], r["flat"]
```

Why it exists, and where it does and does not help:
[The pooling function nobody tunes](https://batchnorm.com).
