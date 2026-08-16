# Method

What is measured, how, and which decisions could have gone another way.

## The cascade

```
video ──► decode at 1 fps ──► SigLIP2 over every frame ──► redundancy collapse
                                        │                         │
                                        └── embeddings ───────────┤
                                                                  ▼
query text ──► SigLIP2 text tower ──► rank ──► select K ──► Qwen2.5-VL ──► answer
```

Stage 1 (decode + encode + collapse) happens **once per video**. Stage 2
(select + VLM) happens **once per query**. That split is the whole economic
argument: the expensive model is 246–809× the cheap one per frame, so the only
thing that matters is how few frames reach it and whether the right ones do.

## Cost model

All measured on one GH200 (aarch64, 64 Grace cores, 96 GB HBM3), not estimated.
Per frame:

| stage | cost/frame | source |
|---|---|---|
| decode 1080p, CPU | 0.25 ms | `bench/decode_resolution_sweep.py` |
| SigLIP2-base-224 | 0.13 ms | `bench/encode_bench.py` |
| SigLIP2-so400m-384 | 1.72 ms | `bench/encode_bench.py` |
| Qwen2.5-VL-7B @64 visual tokens | 32.34 ms | `bench/vlm_bench.py` |
| Qwen2.5-VL-7B @native (~220 tok) | 107.23 ms | `bench/vlm_bench.py` |

The unit of cost throughout the evaluation is **one VLM call**, because on these
numbers everything else rounds to zero.

## Ground truth

A dense Qwen2.5-VL-7B pass over **every** frame of the 4.5 h cab-ride video at
1 fps, for 8 queries: 16,243 frames × 8 = 129,944 VLM scores. This is
simultaneously

- baseline (a), the quality ceiling and the honest cost of the thing everyone
  says is too expensive (2.61 GPU-hours per query per 24 h of video), and
- the reference every cascade result is scored against.

Scoring is `log P("Yes") − log P("No")` at the first generated position: one
forward pass, no sampling, and a continuous score you can threshold anywhere.
Free-text answers would give one bit per frame and no way to draw a curve.

**The VLM is exactly deterministic** given identical input and batching — repeat
runs agree to 0.0000, and batch sizes ≥4 agree exactly (batch 1 differs by up to
0.5 because padding changes). That is what makes the ground truth reproducible
and makes shard merging safe.

## Metrics

**Event recall is the headline**, not frame recall.

Ground-truth positive frames are grouped into events: contiguous runs, merged
across gaps ≤ 3 s (a tunnel briefly occluded for one frame is still one tunnel).
An event counts as found if at least one selected frame inside it is confirmed
by the VLM.

Frame recall is reported too, but it is the wrong headline: it penalises you for
not finding all 40 frames of a 40 s event, which nobody cares about, and it is
capped at `budget / |positives|` whenever the event is long — so it makes every
method look bad for a reason unrelated to selection.

Confirmation is a **lookup** into the ground truth rather than a second VLM call.
That is exact — same model, same frame, same settings — and it is what makes a
4-strategy × 9-budget × 20-seed sweep affordable. The assumption it rests on
(that the indexed frame and the fetched frame are the same image) is checked in
`scripts/verify.py`, and it was false until commit `1a9311d`; see below.

## Selection strategies

All four spend the same budget of VLM calls and differ only in where.

| strategy | what it does |
|---|---|
| `uniform` | evenly spaced over wall-clock time, random phase. What everyone does. |
| `topk` | highest-scoring frames. On real video these are often near-copies of one moment. |
| `nms` | top-k with temporal non-maximum suppression, window = span/(2·budget) |
| `segment` | top-k over redundancy-collapsed segments fixed at index time |
| `segment_adaptive` | the same, but the video is cut into `8 × budget` segments at query time |

`segment_adaptive` exists because the ablation said it should. Sweeping the
index-time `segment_tau` showed the best granularity is not a constant — at 32
calls a coarse segmentation wins, at 128 a finer one does — but the *ratio* of
segments to calls roughly is. Since segmentation is a linear pass over cached
adjacent-frame similarities, granularity can simply be decided per query:
cut at the `n−1` largest frame-to-frame changes, which controls the segment count
exactly and costs O(N log N).

The factor of 8 was chosen from the sweep on this video and then held fixed for
Video-MME. The optimum is broad — 4 to 16 are within noise — which matters more
than its exact value: it is not a knob that needs tuning per video. The extremes
are both degenerate and explain the shape: at factor 1 you take one frame from
every segment, which is uniform sampling in content space; at very large factors
the segments stop constraining anything and it reverts to top-k.

Two of these had bugs that showed up as *flat curves*, which is worth recording
because a flat curve looks like a result:

- NMS with a **fixed** window cannot spend a budget larger than `span/window`.
  Its recall was identical at K=32 and K=128 — an artifact, not a finding. The
  window now adapts to the budget.
- `segment` originally topped up a budget larger than the segment count with a
  global top-k, which re-introduced exactly the redundancy the segmentation had
  removed. It now takes frames round-robin across ranked segments.

Selection records how many candidates it actually produced, so a saturated
strategy is labelled rather than silently plotted as a plateau.

## Query surface forms

Each query exists in two forms (`configs/queries_glasgow.json`):

- a **caption** for SigLIP, which is trained on image captions
- a **yes/no question** for the VLM, which is instruction-tuned

Handing SigLIP the question form measurably handicaps every retrieval-based
strategy and would flatter uniform sampling for a reason that has nothing to do
with selection. The captions were written once, before any recall number was
computed, and were not revised in response to results. `--retrieval-form question`
exists so the size of that handicap can be measured rather than hidden.

## Variance

- `uniform` has a random phase, so it gets N independent seeds and we report the
  spread across them. This matters: uniform sampling's luck is precisely why it
  is a stronger baseline than people expect.
- The index-based strategies are **deterministic** given the index. Their
  variance is across *queries*, reported as a bootstrap CI. We do not manufacture
  seed variance for them by adding noise.

## The bug the cross-check exists for

The ablation and the main recall curve are two paths to the same number, so they
were run against the same ground truth and the same index and compared. They
disagreed by **12×**. Two causes, both silent:

- `ablate.py` handed SigLIP the ground truth's yes/no *question* while
  `eval_recall_curve.py` handed it the *caption*. That difference alone is
  0.021 → 0.261 event recall at 8 VLM calls, which is also the cleanest available
  measurement of how much the surface form matters.
- `ablate.py` did not restrict the index to the ground truth's time range. Against
  a partial ground-truth run, every selection past its end was snapped onto the
  last covered frame and scored as though it had been there.

`evaluate_selection` now raises when a selection is more than 1.5 s from any
ground-truth frame instead of snapping. A wrong number that looks plausible is
worse than a crash.

## The bug that verify.py exists for

`FrameStream(t)` and `FrameFetcher(t)` returned **different images** for the same
`t` — mean 7.6 grey levels apart, max 31.9.

ffmpeg's `fps=` filter re-stamps its output onto a nominal grid, so its
timestamps do not identify which source frame it picked. At 25 fps sampled to
1 fps the chosen frame can sit up to half a second from where its timestamp
claims — a completely different picture on moving footage.

The fix is to select on elapsed source time, which preserves the original PTS:

```
select='isnan(prev_selected_t)+gte(t-prev_selected_t\,1)'
```

reading the true PTS back via `showinfo` (2% overhead), and seeking half a
source-frame early in the fetcher so float rounding cannot land on the neighbour.
Frames are now bit-identical between the two paths (diff exactly 0.000).

This invalidated a ground-truth run that was 25% complete and every index built
before it. Both were rebuilt.

## Analyses added after the first pass

`scripts/analysis.py` holds four derivations and their checks against the
measurements. Each exists because a number alone was not enough to act on.

- **Uniform sampling in closed form.** `P(find event) = min(1, K·L/N)` per event.
  Predicts 20 seeded runs to within 10% at every budget above 16; the residual is
  event clustering, which the formula does not model.
- **Cascade speedup.** With cost ratio `R = e/c` and filter ratio `F = N/K`,
  `1/S = 1/R + 1/F`. Amortised over Q queries, `1/S = 1/(R·Q) + 1/F`, so `S → F`.
- **The selector's ceiling.** The best rank the cheap stage gives any frame inside
  each ground-truth event bounds top-k exactly. Averaged per query, then across
  queries, to match how event recall is computed — pooling all events instead
  weights queries by how many they happen to have and is not comparable.
- **Oracle-confidence stratification.** Recall recomputed with a floor on each
  event's peak ground-truth score. This is the analysis that changed the
  project's conclusion, and it is validated by inspecting sampled frames from
  both ends of the confidence range rather than by correlation alone.

## Benchmarks

Two, under their own published protocols, chosen for different reasons.

- **Video-MME (long split)** because the frame-selection literature reports on
  it. 900 questions, K frames per question, four-way multiple choice, no
  subtitles. Result: no strategy beats uniform outside the noise.
- **MomentSeeker (t2v split)** because its task is the one this system performs.
  Fixed 10-second candidate chunks, IoU threshold 0.3, R@1 and mAP@5. The paper
  and its released code define mAP@5 differently; both are computed.

Baseline cost is measured rather than quoted, in a separate virtualenv because
`languagebind` pins `transformers<5` and installing it into the main environment
silently downgrades it. InternVideo2's released checkpoint requires its training
repo, so its vision tower is reconstructed from the published architecture and
labelled as a compute estimate rather than that model.

## Choices that could have gone differently

- **1 fps sampling.** Events shorter than a second can be missed by *every*
  method here, including the ground truth. The cost model scales linearly with
  this, so a higher rate is affordable — it is a knob, not a limit.
- **`segment_tau = 0.90`** was fixed on the cab-ride video and held constant on
  the benchmark. The ablation sweeps it; it was not tuned on Video-MME.
- **Native visual-token budget for ground truth.** At 64 tokens the VLM agrees
  on the *decision* almost perfectly (AUC 0.966–0.998) but finds only half the
  positives on rare queries (0.020 vs 0.040 positive rate for "another train"),
  so ground truth pays for native resolution.
- **h264 only.** HEVC and AV1 decode meaningfully slower on CPU; the decode
  headroom reported here is for h264.
