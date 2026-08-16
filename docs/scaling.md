# From one video to a library

framesieve has two ways to hold an index, and you should switch between them on
purpose rather than by accident.

| | `VideoIndex` | `Collection` |
|---|---|---|
| holds | one video | any number of videos |
| stored | a `.npz` sidecar, loaded into RAM | LanceDB, on disk |
| search | exact, a matrix multiply over every frame | approximate, a vector index |
| answers | *when* in this video | *which* video, and when |
| needs | nothing beyond framesieve | `pip install "framesieve[collection]"` |
| good up to | a few hundred hours | tens of thousands |

## When to switch

The deciding number is RAM, and it is easy to work out. At 1 fps a video-hour is
3,600 frames, and each frame is a 768-dimension float32 vector — 3 KB. So:

| footage | vectors | as a numpy array |
|---|---|---|
| 1 hour | 3,600 | 11 MB |
| 100 hours | 360,000 | 1.1 GB |
| **500 hours** | 1.8M | **5.5 GB** |
| 1,000 hours | 3.6M | 11 GB |
| 10,000 hours | 36M | 110 GB |

**Under a few hundred hours, stay on `VideoIndex`.** It is exact, it needs no
extra dependency, and a search is a matrix multiply: 6 ms over a 4.5-hour video,
and still only a few hundred ms over a hundred hours.

**Switch when the corpus stops fitting comfortably in memory**, or as soon as you
need to search *across* videos rather than within one — `Collection` answers
"which of my 4,000 recordings shows this", and no per-video index can.

Measured on 10 million vectors — 2,778 video-hours, 62 GB of vectors and index:

| | |
|---|---|
| open the collection | 0.18 GB resident |
| search | 112 ms median, 139 ms p90 |
| peak resident over 30 queries | 5.5 GB |
| smallest memory cap that works | **8 GB** (4 GB is OOM-killed) |
| the same vectors as a numpy array | 31 GB, always |

It is about 6× less memory, not constant memory: graph traversal has a real
working set. But it is the difference between that corpus running on an 8 GB
machine and not running at all.

## Switching

You do not re-encode anything. Indexing is the expensive half, it is per-video,
and it has already happened — a `Collection` is built by merging the sidecars
you have.

```python
import framesieve as fs

lib = fs.Collection("library.lancedb")

# whatever you already indexed, however you indexed it
lib.add_indexes("footage/*.npz")

# or index new videos straight into it
lib.add("new_camera.mp4")

lib.build_ann()          # once, after the bulk load
```

Because indexing is per-video it also parallelises without any coordination:
index on as many machines as you like, copy the `.npz` files to one place, and
merge them.

```python
for hit in lib.search("a red car", k=20):
    print(hit.video, hit.timecode, hit.score)
```

### Going back the other way

A `Collection` is not a replacement for `VideoIndex`, and the two compose. The
usual shape is a broad search across the library to find *which* video, then the
per-video index to spend expensive-model calls inside it:

```python
hit = lib.search("a red car", k=5, per_video=1)[0]      # which recording
video = fs.open(hit.video)                              # then work inside it
best = video.search("a red car", k=32, confirm=True)    # with the VLM
```

`Collection` deliberately does not carry the frame-selection strategies. Those
are about spending a budget well inside one video, which is a different problem
from finding the video.

## Choosing the index

`build_ann()` defaults to `IvfHnswFlat`, which is not the usual advice, and the
measurements are why.

**The quantized index types do not work on these embeddings.** The best
similarity across 205 hours of video is 0.16, and neighbouring frames differ in
the third decimal. Quantization error is larger than the signal being ranked:

| index | size | recall@20 |
|---|---|---|
| IvfPq | 78 MB | **0%** |
| IvfRq | 86 MB | 24% |

That holds at every probe count up to scanning all partitions, and with
`refine_factor`. It is a property of the embeddings, not of LanceDB, and it is
worth knowing before reaching for the standard IVF_PQ recipe.

Among the ones that do work, judge them on whether the **top hit** matches an
exact scan, not on recall@20 — recall@20 asks whether the same twenty rows come
back, and it ranks these backwards. Over 15 queries with an exact scan as the
answer key:

| nprobes | 20 | 50 | 100 | 200 | 400 |
|---|---|---|---|---|---|
| `IvfFlat` | 6/15 | 9/15 | 11/15 | 12/15 | 15/15 at 81 ms |
| **`IvfHnswFlat`** | 14/15 | **15/15 at 10 ms** | 15/15 | 15/15 | 15/15 |

IVF only matches the graph by probing half its partitions, for 8× the latency,
because partition scanning grows with `nprobes` and graph traversal does not.

```python
lib.build_ann("hnsw")       # default: best answer per millisecond
lib.build_ann("hnsw_sq")    # a quarter of the disk, a few points of recall
lib.build_ann("flat")       # exact within probed partitions; raise nprobes
```

Check what yours costs rather than trusting this table — the right setting
depends on how your embeddings are distributed:

```python
lib.recall_at(["a red car", "someone running", "an empty room"], k=20)
```

## Things that will surprise you

- **Consecutive frames are near-identical.** At 1 fps an uncollapsed top-5 is one
  moment returned five times. `search()` collapses runs by default
  (`min_gap_s=30`); `per_video=1` goes further and returns one hit per video.
- **Similarity has no absolute scale.** 0.16 is an excellent match for SigLIP and
  0.05 is nothing. Compare within a query, never across.
- **The first query in a process is slow** — it loads the text encoder, about
  4 s. Everything after that is the number in the tables above.
- **`build_ann()` is not automatic.** Until you call it, every search scans the
  whole table: correct, and linearly slower as the corpus grows.
- **Adding rows after building the index** leaves them unindexed until you build
  again. `list_indices()` reports `num_unindexed_rows`.
