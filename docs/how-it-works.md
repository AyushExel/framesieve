# How it works

The full argument, with the figures and the things that failed, is in
[Search a day of video in a second](https://batchnorm.com). This is the short version for
someone deciding whether the design fits their problem.

## The shape

```
video ─► decode @1fps ─► small image encoder on every frame ─► collapse repeats
                                    │                               │
                                    └──────── embeddings ───────────┤
                                                                    ▼
query ─► encode text ─► rank every frame ─► pick K ─► VLM on those K ─► answer
```

Two stages, and the whole design follows from one asymmetry: **the cheap pass
runs once per video, the expensive pass runs once per query.**

## The numbers that force it

Measured on one GH200, per frame:

| stage | cost/frame | per 24 h of video @ 1 fps |
|---|---|---|
| decode 1080p (CPU) | 0.25 ms | 9 min |
| SigLIP2-base-224 — the cheap pass | 0.13 ms | **11 seconds** |
| Qwen2.5-VL-7B at native resolution — the expensive pass | 107 ms | **2.61 GPU-hours** |

The expensive model costs **809×** the cheap one. That ratio is the entire
reason this library exists: at 809× you cannot afford to look at everything, and
the only question left is which frames you look at.

## What a cascade can buy

Let `c` be the cheap model's cost per frame, `e` the expensive one's, `N` the
frames and `K` the survivors:

```
dense    = N·e
cascade  = N·c + K·e

S = N·e / (N·c + K·e) = 1 / (c/e + K/N)
```

Name the two ratios — `R = e/c` the **cost ratio**, `F = N/K` the **filter
ratio** — and it collapses to

```
1/S = 1/R + 1/F        so        S = R·F / (R + F)
```

The reciprocals add: the two ceilings combine **like parallel resistors**. This
is the most portable thing in the design and the thing people most often get
wrong. If your cheap model is 1000× cheaper but you only filter 10×, your
speedup is 9.9×, and making the cheap model cheaper buys *nothing*. Filtering
harder buys everything.

Index once and serve `Q` queries and the cheap stage's share divides by Q:

```
1/S = 1/(R·Q) + 1/F     →     S → F   as Q grows
```

Past a handful of queries the cost ratio drops out entirely and the filter ratio
is the only ceiling left. That is the argument for building an index instead of
filtering on the fly, in one line.

## Why the index is small

An hour of video at 1 fps is 3,600 frames × 768 dimensions in float32, about
11 MB, stored as a Lance dataset. It is kept at the precision the ranking runs
at, so nothing has to be cast at query time.

That also means an index is **portable and cheap to read**: no GPU, no model, no
torch. You can build indexes on a GPU box and query them anywhere.

## Why decode is not the problem

The received wisdom is that CPU decode runs at a few hundred frames per second,
so a day of video costs hours before a model sees anything. Measured, that is off
by 30–70×:

```
resolution     backend   frame/s      xRT   index a 24h video in
640x480        cpu        10,869    435x    0.06 h
1920x1080      cpu         4,041    162x    0.15 h
3840x2160      cpu         1,886     75x    0.32 h
1920x1080      nvdec         707     28x    0.85 h
```

GPU decode is **slower** than CPU here in wall clock. Its advantage is that it
costs about 0.5 CPU cores instead of 16, which matters only when the CPU has
something else to do. See [METHOD.md](METHOD.md) for the traps, including NVDEC
missing from a driver install and failing silently.

## Why candidate selection is not just top-k

Real footage repeats itself. Take the `k` highest-scoring frames of a long video
and you often get `k` near-copies of one moment, which spends the whole budget
learning one thing.

`segment_adaptive`, the default, cuts the video into `8 × budget` segments at the
points where the picture changes most, then takes the best frame from each of the
top segments. Measured against dense ground truth, diversity-aware selection is
worth **+3.2 points at 8 calls, +11.7 at 128, +0.8 at 1,024** — an inverted U. At
small budgets there is no room to diversify; at large ones you have already taken
everything.

Plain top-k, meanwhile, already achieves the ceiling its ranking allows to within
**0.2 points** at every budget. There is no slack in the obvious algorithm; the
only thing left is diversity.

## Why the frames in a segment become one score with `mean of top k`

The obvious way to score a candidate segment is `max` over the frames in it —
"does this segment contain a matching frame?". Measured, that is the wrong end of
a dial: replacing it with the mean of the top 4 is worth **+2.6 R@1** on
MomentSeeker for no extra compute, which is what takes the 93M-parameter
retrieval stage past a billion-parameter video model.

That result turned out not to be about video at all, and is its own piece:
[The pooling function nobody tunes](https://batchnorm.com). The reusable part is
`framesieve.pooling`.

## What it is not

- **Not a video question-answering system.** It finds *where*, not *what
  happened overall*. On whole-video questions ("how many times does X occur")
  every selection strategy landed inside every other's confidence interval, and
  the frame budget dominated.
- **The frame encoder cannot hear.** `--audio` covers that: it transcribes
  with Whisper and indexes what was said as a separate signal.
- **A 224-pixel embedding cannot read a sign.** `--ocr` covers that: it reads
  the text on screen and indexes it, also as its own signal.
- **Not exhaustive.** At 32 model calls on a 4.5-hour video it finds 23% of
  labelled events, against uniform sampling's 0.9%. It is 26× better than the
  default, not complete. Recall against compute is a curve, and there is no point
  on it where you are finished.
