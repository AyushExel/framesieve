# Quickstart

## Install

`torch` first, from the index for your platform. Installing it as a transitive
dependency is the most common way to end up on a CPU-only wheel, which is silent
and about 30× slower.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128   # your CUDA
pip install framesieve
```

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
framesieve search holiday.mp4 "a dark tunnel" --no-refine
```

`index` writes a sidecar next to the video — about **5 MB and 15 seconds per hour
of footage** — and you run it once. Everything after that reads the sidecar.

`search` ranks every indexed frame against your text and returns the best
candidates. `--no-refine` keeps it to retrieval only, which is roughly a
millisecond and touches no pixels.

Drop `--no-refine` and a vision-language model actually looks at each candidate
and says yes or no:

```bash
framesieve search holiday.mp4 "a dark tunnel" -k 16
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
curve = fs.load("holiday.npz").score(query_vector)   # numpy only
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
- **Anything audio.** No speech, no sound. Frames only.
- **Whole-video questions.** "How many times does X happen?" needs coverage, not
  relevance; on Video-MME every selection strategy landed inside every other's
  confidence interval, and the frame budget dominated.

## Next

- [How it works](how-it-works.md) — the cascade, the cost model, and why the
  index is small
- [API reference](api.md)
- [Search a day of video in a second](https://batchnorm.com) — the measurements behind all
  of this, including what failed
