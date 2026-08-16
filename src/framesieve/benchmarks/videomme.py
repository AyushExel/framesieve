"""Video-MME under its standard protocol.

The protocol, matched to what the frame-selection literature reports:
  - multiple choice, four options, accuracy is the metric
  - no subtitles (the "w/o subs" column), because we are evaluating *visual*
    frame selection and subtitles would let the model answer without looking
  - a fixed frame budget K, uniformly sampled unless a selector says otherwise
  - answers parsed as a single letter A-D

We report the long split separately and headline it, because that is the split
whose videos (30-60 min) are closest to the regime this project cares about --
and because averaging it with 1-minute clips would hide exactly the effect we
are trying to measure.

Nothing here tunes on the benchmark. The selector's hyperparameters come from
the held-out cab-ride video; this module only measures.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

LETTERS = ["A", "B", "C", "D"]


@dataclass
class VmmeItem:
    video_id: str
    videoID: str
    duration: str          # short | medium | long
    domain: str
    task_type: str
    question_id: str
    question: str
    options: list[str]
    answer: str            # gold letter
    video_path: str | None = None

    def prompt(self) -> str:
        """The standard Video-MME prompt format."""
        opts = "\n".join(self.options)
        return (f"{self.question}\n{opts}\n"
                "Answer with the option's letter from the given choices directly.")

    def retrieval_query(self) -> str:
        """A text query for the cheap encoder.

        The question alone is what a retrieval system would be handed, so that is
        what we use. Folding the options in would leak the answer set into
        selection and make the comparison to uniform sampling dishonest.
        """
        return self.question


def load_items(parquet_path: str, video_dir: str,
               durations: Sequence[str] = ("long",)) -> list[VmmeItem]:
    import pyarrow.parquet as pq

    df = pq.read_table(parquet_path).to_pandas()
    df = df[df["duration"].isin(list(durations))]
    items: list[VmmeItem] = []
    missing = 0
    for _, r in df.iterrows():
        path = os.path.join(video_dir, f"{r['videoID']}.mp4")
        if not os.path.exists(path):
            missing += 1
            path = None
        items.append(VmmeItem(
            video_id=str(r["video_id"]), videoID=str(r["videoID"]),
            duration=str(r["duration"]), domain=str(r["domain"]),
            task_type=str(r["task_type"]), question_id=str(r["question_id"]),
            question=str(r["question"]), options=[str(x) for x in r["options"]],
            answer=str(r["answer"]).strip().upper()[:1], video_path=path))
    if missing:
        print(f"  warning: {missing}/{len(items)} questions have no local video file")
    return items


_LETTER_RE = re.compile(r"\b([ABCD])\b")


def parse_answer(text: str) -> str | None:
    """Extract the chosen letter. Returns None if the model refused to choose,
    which is counted as wrong rather than silently dropped."""
    t = text.strip()
    if not t:
        return None
    if t[0].upper() in LETTERS and (len(t) == 1 or not t[1].isalnum()):
        return t[0].upper()
    m = _LETTER_RE.search(t.upper())
    return m.group(1) if m else None


def accuracy(preds: Sequence[str | None], golds: Sequence[str]) -> float:
    ok = sum(1 for p, g in zip(preds, golds) if p is not None and p == g)
    return ok / max(1, len(golds))


def accuracy_by(items: Sequence[VmmeItem], preds: Sequence[str | None],
                key: str) -> dict[str, tuple[float, int]]:
    buckets: dict[str, list[tuple[str | None, str]]] = {}
    for it, p in zip(items, preds):
        buckets.setdefault(getattr(it, key), []).append((p, it.answer))
    return {k: (accuracy([p for p, _ in v], [g for _, g in v]), len(v))
            for k, v in sorted(buckets.items())}


def bootstrap_accuracy_ci(preds: Sequence[str | None], golds: Sequence[str],
                          n_boot: int = 2000, alpha: float = 0.05,
                          seed: int = 0) -> tuple[float, float, float]:
    """Accuracy with a percentile CI over questions.

    Video-MME's long split is 900 questions; a 1-point difference there is within
    noise, and reporting a bare point estimate would invite over-reading it.
    """
    correct = np.array([1.0 if (p is not None and p == g) else 0.0
                        for p, g in zip(preds, golds)])
    if len(correct) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = rng.choice(correct, size=(n_boot, len(correct)), replace=True).mean(axis=1)
    return (float(correct.mean()), float(np.percentile(boots, 100 * alpha / 2)),
            float(np.percentile(boots, 100 * (1 - alpha / 2))))
