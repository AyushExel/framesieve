# Quickstart

## Install

`torch` first, from the index for your platform. If you have a GPU, installing
torch as a transitive dependency is the most common way to end up on a CPU-only
wheel by accident — silent, and much slower than the card you paid for.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128   # your CUDA
pip install framesieve
```

**No GPU required.** Everything picks CUDA if there is one, Apple silicon if
there is one, and CPU otherwise — no flags. On CPU, indexing an hour of video
takes a minute or two instead of fifteen seconds; a search is ~110 ms instead
of ~6 ms, because the cost is encoding the query text, not the ranking. The one
part that really wants a GPU is `--confirm`, which runs a 7B vision-language
model.

You also need `ffmpeg` on your `PATH`, with h.264 support:

```bash
ffmpeg -hide_banner -decoders | grep h264      # should print at least one
```

For the `--confirm` stage, which runs a vision-language model on the frames that
survive retrieval:

```bash
pip install "framesieve[vlm]"
```

## Two commands

```bash
framesieve index  holiday.mp4
framesieve search holiday.mp4 "a dark tunnel"
```

`index` writes a sidecar next to the video — about **11 MB and 15 seconds per hour
of footage** — and you run it once. Everything after that reads the sidecar.

`search` ranks every indexed frame against your text and returns the best
candidates. That is retrieval only and touches no pixels: warm, on a GPU, about
6 ms, most of it encoding the query text; the ranking itself is about a
millisecond.

Add `--confirm` and a vision-language model actually looks at each candidate
and says yes or no:

```bash
framesieve search holiday.mp4 "a dark tunnel" -k 16 --confirm
```

```
query    : 'a dark tunnel'
index    : 16,244 frames, 4.51 h, siglip2-base-224
strategy : segment_adaptive, 16 candidates (0.10% of frames)
timing   : select 1.5 ms, fetch 0.19 s, vlm 0.24 s

7 hit(s) above threshold 0.0:
          time    hh:mm:ss     vlm score   similarity
        4821.0     1:20:21          8.50        0.147
        4822.0     1:20:22          7.88        0.146
        ...
```

The **vlm score** is a log-odds margin: 0 is a coin flip, +2 is about 7:1 for
yes, −2 the reverse. The **similarity** is the retrieval score, which is only
comparable within one query — do not threshold it.

## From Python

```python
import framesieve as fs

video = fs.open("holiday.mp4")          # indexes if needed, loads if not
print(video)                            # <VideoIndex holiday.mp4 16,244 frames, 4.51 h>

for hit in video.search("a dark tunnel", k=8):
    print(hit.timecode, round(hit.score, 3))

# ask a real model, then keep only what it confirmed
hits = video.search("a dark tunnel", k=8, confirm=True)
for hit in hits.above(0.0):
    print(hit.timecode, hit.vlm_score)

# the raw material: similarity for every frame, aligned with video.times
curve = video.score("a dark tunnel")

# and the actual pixels
frames = video.frames(hits[:4])         # list of uint8 HWC arrays
```

`import framesieve` costs about 50 ms and does **not** import torch. Building an
index needs torch; reading one does not, so you can query and inspect indexes on
a machine with no GPU stack at all — and `score()` accepts a precomputed query
vector, so ranking works there too:

```python
curve = fs.load("holiday.lance").score(query_vector)   # numpy only
```

## Phrase queries as captions

The retrieval encoder is caption-trained. `"a dark tunnel"` works; `"is the train
in a tunnel?"` works measurably worse, because the question form is not what the
encoder saw during training. On MomentSeeker the difference between caption-form
and question-form queries is **24.8 against 15.3 R@1**.

The vision-language model is the opposite — it wants a question — so
`search()` builds one for it automatically (`"Does this frame show: {query}?"`).
Override it with `question=` if you want something more specific.

## Choosing k

`k` is how many candidate frames you consider, and with `confirm=True` how many
model calls you spend. Measured against dense ground truth on a 4.5-hour video:

| k | share of frames | event recall | wall clock with `confirm` |
|---|---|---|---|
| 8 | 0.05% | 8.8% | ~0.3 s |
| 32 | 0.20% | 23.2% | ~1.1 s |
| 128 | 0.79% | 43.6% | ~4 s |
| 512 | 3.2% | 56.7% | ~17 s |

Recall keeps climbing and so does the bill; there is no k at which you have
"finished". Start at 32.

## Where it will not help

- **Text on signs.** The retrieval stage is a 224-pixel global embedding and
  cannot read. On MomentSeeker's OCR split it scores R@1 3.4.
- **Where in the frame.** "the cup on the left" scores 3.8. Global embeddings
  have no spatial handle.
- **Whole-video questions.** "How many times does X happen?" needs coverage, not
  relevance; on Video-MME every selection strategy landed inside every other's
  confidence interval, and the frame budget dominated.

## Speech and on-screen text

Frames are not the only signal in a video. `--audio` transcribes with Whisper;
`--ocr` reads the text on screen. Both index timed spans that become searchable
alongside the frames:

```bash
pip install "framesieve[audio,ocr]"
framesieve index talk.mp4 --audio --ocr
framesieve search talk.mp4 "the part about pricing" --source speech
```

```python
video = fs.open("talk.mp4")
for hit in video.search("a drone flying"):      # both modalities, merged
    print(hit.timecode, hit.source, hit.text or "")
```

`source=` restricts to one of `"visual"`, `"speech"`, `"text"`. They are never
ranked against each other — a frame similarity, a spoken sentence and a line of
on-screen text are different quantities — but when several point at the same
moment it comes back once naming all of them, which is the strongest signal any
of them can give you.

## More than one video

Everything above is one video, held in memory. That is right up to a few hundred
hours; past that, or as soon as you want to search *across* recordings rather
than within one, move to a `Collection` — the same vectors in LanceDB on disk:

```python
lib = fs.Collection("library.lancedb")
lib.add_indexes("footage/*.lance")   # merge indexes you already built
lib.build_ann()

for hit in lib.search("a red car", k=20):
    print(hit.video, hit.timecode, hit.score)
```

You do not re-encode anything to switch, and speech/OCR sidecars matching the
glob are skipped automatically. See
**[Scaling to a library](scaling.md)** for where the threshold is, what it costs,
and which index type to use — the usual `IVF_PQ` advice does not work here.

## Next

- [How it works](how-it-works.md) — the cascade, the cost model, and why the
  index is small
- [API reference](api.md)
- [Scaling to a library](scaling.md) — many videos, on disk
- [Search a day of video in a second](https://batchnorm.com) — the measurements behind all
  of this, including what failed
