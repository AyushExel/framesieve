"""Build the images the README leads with.

Two of them, both from artefacts already in the repo rather than mocked up:

    banner      what the tool does, in one picture: a query, the frames it
                found in 4.5 hours of footage, and where it looked compared
                with the default approach
    at_a_glance the three numbers that matter, as one strip

Both render light and dark, and the README picks between them with a <picture>
element so neither theme gets the other's background.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from framesieve.figures import THEME, save  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    d = json.load(open(os.path.join(ROOT, "runs/hero_demo.json")))
    frames = sorted(glob.glob(os.path.join(ROOT, "figures/hero_frames/*.jpg")))
    return d, frames


HERO = [
    # query as a user would type it, the timestamp it returned, and the
    # expensive model's verdict on that frame. All from one 4.5-hour video.
    ("a station platform", 1.0, 9.87),
    ("a stone viaduct", 14777.0, 7.75),
    ("the sea", 16069.0, 7.87),
    ("a red signal light", 11747.0, 7.50),
]


def _tc(t: float) -> str:
    t = int(t)
    return f"{t//3600}:{t%3600//60:02d}:{t%60:02d}"


def banner(mode: str = "light") -> str:
    """Four different searches of one long video, and what each returned.

    One query would show a lucky match; four unrelated ones show the thing is
    general, which is the actual claim. The frames are real output and the
    scores beside them are the vision-language model's verdict on that frame,
    so a reader can check the labels against the pictures.
    """
    t = THEME[mode]
    paths = sorted(glob.glob(os.path.join(ROOT, "figures/hero_queries/*.jpg")))
    by_key = {os.path.basename(p).split("_", 1)[1].rsplit(".", 1)[0]: p
              for p in paths}

    fig = plt.figure(figsize=(13.0, 4.55))
    fig.patch.set_facecolor(t["surface"])
    gs = fig.add_gridspec(3, 4, height_ratios=[0.52, 0.20, 1.55],
                          hspace=0.10, wspace=0.045,
                          left=0.028, right=0.972, top=0.965, bottom=0.075)

    ax = fig.add_subplot(gs[0, :]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0, 0.62, "framesieve", fontsize=27, fontweight="600",
            color=t["ink"], va="center", family="monospace")
    ax.text(0, 0.08, "find things in video by describing them",
            fontsize=13.5, color=t["ink2"], va="center")
    ax.text(1.0, 0.64, "one 4.5-hour video, indexed once", fontsize=12.5,
            color=t["ink2"], ha="right", va="center")
    ax.text(1.0, 0.12, "four searches  ·  25 ms each", fontsize=12.5,
            color=t["series"][0], ha="right", va="center", fontweight="600",
            family="monospace")

    for i, (query, ts, score) in enumerate(HERO):
        lab = fig.add_subplot(gs[1, i]); lab.axis("off")
        lab.set_xlim(0, 1); lab.set_ylim(0, 1)
        lab.text(0.5, 0.42, f'"{query}"', fontsize=13, color=t["ink"],
                 ha="center", va="center", family="monospace")

        a_ = fig.add_subplot(gs[2, i])
        key = query.replace(" ", "_")
        a_.imshow(plt.imread(by_key[key]))
        a_.set_xticks([]); a_.set_yticks([])
        for sp in a_.spines.values():
            sp.set_edgecolor(t["series"][0]); sp.set_linewidth(2.0)
        a_.set_xlabel(f"{_tc(ts)}      confirmed {score:+.2f}", fontsize=10.5,
                      color=t["ink2"], family="monospace", labelpad=5)

    fig.text(0.028, 0.012,
             "Real output. The score is the vision-language model's verdict on "
             "that frame: 0 is a coin flip, +2 is about 7:1 for yes.",
             fontsize=10, color=t["ink3"])
    return save(fig, os.path.join(ROOT, "figures/banner.png"), mode)


def coverage(mode: str = "light") -> str:
    """Where the budget went, against sampling every Nth frame.

    The banner sells the capability; this is the evidence for it. Every grey bar
    is a tunnel the ground truth marks, and both rows spent the same 32 model
    calls.
    """
    d, _ = _load()
    t = THEME[mode]
    hero, picks = d["hero"], d["picks"]
    events = hero["events"]
    duration = 16244.0

    fig, ax = plt.subplots(figsize=(13.0, 2.05))
    fig.patch.set_facecolor(t["surface"])
    ax.set_xlim(0, duration); ax.set_ylim(-0.2, 1.55); ax.axis("off")
    ax.text(0, 1.42, 'searching 4.5 hours for  "a dark tunnel"',
            fontsize=12.5, color=t["ink"], va="center", family="monospace")

    for a, b, _c in events:
        ax.plot([a, max(b, a + 12)], [0.92, 0.92], color=t["ink3"],
                linewidth=4, solid_capstyle="butt", zorder=2)
    ax.text(duration * 1.005, 0.92, f"{len(events)} tunnels", fontsize=10.5,
            color=t["ink3"], va="center")

    def row(times, y, colour, name, emphasise):
        """Draw where one strategy looked, and count what it hit.

        Counts are computed rather than written in: a figure claiming "0 found"
        beside a visible hit is the fastest way to lose a reader.
        """
        ax.plot([0, duration], [y, y], color=t["grid"], linewidth=1, zorder=1)
        on = [x for x in times if any(a - 2 <= x <= b + 2 for a, b, _ in events)]
        off = [x for x in times if x not in on]
        ax.scatter(off, [y] * len(off), s=17, color=colour, zorder=3, linewidths=0)
        if on:
            ax.scatter(on, [y] * len(on), s=95, marker="*", color=colour,
                       zorder=4, linewidths=0)
        hit = sum(1 for a, b, _ in events
                  if any(a - 2 <= x <= b + 2 for x in times))
        ax.text(duration * 1.005, y, f"{name}: {hit} of {len(events)}",
                fontsize=10.5, color=colour, va="center",
                fontweight="600" if emphasise else "normal")
        return hit

    row(picks["uniform"], 0.45, t["muted"], "every Nth frame", False)
    row(picks.get("segment_adaptive", picks.get("segment", [])), 0.06,
        t["series"][0], "framesieve", True)

    fig.text(0.028, 0.02,
             "Both rows spent the same 32 model calls. Over 200 random offsets, "
             f"sampling every Nth frame finds nothing at all "
             f"{100*hero['uniform_miss_rate']:.0f}% of the time.",
             fontsize=10, color=t["ink3"])
    fig.subplots_adjust(left=0.028, right=0.90, top=0.94, bottom=0.16)
    return save(fig, os.path.join(ROOT, "figures/coverage.png"), mode)


def at_a_glance(mode: str = "light") -> str:
    """Three measured numbers, as a strip, with what each is against."""
    t = THEME[mode]
    cards = [
        ("15 s", "to index an hour of video", "and 5 MB, measured over 205 h"),
        ("6 ms", "to search all 4.5 hours", "on a GPU; about 110 ms on CPU"),
        ("26×", "what sampling every Nth frame finds", "at the same 32 model calls"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 2.05))
    fig.patch.set_facecolor(t["surface"])
    for ax, (big, mid, small) in zip(axes, cards):
        ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.add_patch(FancyBboxPatch(
            (0.02, 0.06), 0.96, 0.88, boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor=t["surface"], edgecolor=t["grid"], linewidth=1.4,
            transform=ax.transAxes, zorder=0))
        ax.text(0.075, 0.66, big, fontsize=31, fontweight="600",
                color=t["series"][0], va="center", family="monospace")
        ax.text(0.075, 0.35, mid, fontsize=12.5, color=t["ink"], va="center")
        ax.text(0.075, 0.17, small, fontsize=10.5, color=t["ink3"], va="center")
    fig.subplots_adjust(left=0.012, right=0.988, top=0.96, bottom=0.04,
                        wspace=0.045)
    return save(fig, os.path.join(ROOT, "figures/at_a_glance.png"), mode)


if __name__ == "__main__":
    for mode in ("light", "dark"):
        print(banner(mode))
        print(coverage(mode))
        print(at_a_glance(mode))
