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


def banner(mode: str = "light") -> str:
    """Query in, frames out, and where each approach spent its budget.

    The frames are real output for the caption shown, and the timeline below
    them is the same run: every grey tick is a labelled tunnel, and the two rows
    of dots are where the 32 model calls actually went. Uniform's row landing on
    no tick is the entire argument for the project, so it is drawn rather than
    asserted.
    """
    d, frame_paths = _load()
    t = THEME[mode]
    hero, picks = d["hero"], d["picks"]
    events = hero["events"]
    duration = 16244.0

    # six frames spread across the run, so the strip reads as "all through the
    # video" rather than "one dense patch"
    pick_idx = np.linspace(0, len(frame_paths) - 1, 6).round().astype(int)
    chosen = [frame_paths[i] for i in dict.fromkeys(pick_idx)]

    fig = plt.figure(figsize=(13.0, 5.35))
    fig.patch.set_facecolor(t["surface"])
    gs = fig.add_gridspec(3, len(chosen), height_ratios=[0.40, 1.65, 0.62],
                          hspace=0.14, wspace=0.035,
                          left=0.035, right=0.965, top=0.97, bottom=0.06)

    # --- title band -------------------------------------------------------
    ax = fig.add_subplot(gs[0, :]); ax.axis("off")
    ax.text(0, 0.60, "framesieve", fontsize=27, fontweight="600",
            color=t["ink"], va="center", family="monospace")
    ax.text(0, 0.05, "search long video without running a VLM on every frame",
            fontsize=13, color=t["ink2"], va="center")
    ax.text(1.0, 0.62, "4.5 hours of footage", fontsize=12.5, color=t["ink2"],
            ha="right", va="center")
    ax.text(1.0, 0.10, "32 model calls  ·  25 ms to search", fontsize=12.5,
            color=t["series"][0], ha="right", va="center", fontweight="600",
            family="monospace")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # --- the frames it found ---------------------------------------------
    for i, p in enumerate(chosen):
        a = fig.add_subplot(gs[1, i])
        a.imshow(plt.imread(p))
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_edgecolor(t["series"][0]); s.set_linewidth(2.0)
        secs = int(os.path.basename(p).split("_t")[1].split(".")[0])
        a.set_xlabel(f"{secs//3600}:{secs%3600//60:02d}:{secs%60:02d}",
                     fontsize=10.5, color=t["ink2"], family="monospace",
                     labelpad=4)

    # --- where each approach looked --------------------------------------
    ax = fig.add_subplot(gs[2, :])
    ax.set_xlim(0, duration); ax.set_ylim(-0.15, 1.5)
    ax.axis("off")
    ax.text(0, 1.36, 'query:  "a view from inside a dark railway tunnel"',
            fontsize=12, color=t["ink"], va="center", family="monospace")

    for a, b, _ in events:                    # every labelled tunnel
        ax.plot([a, max(b, a + 12)], [0.86, 0.86], color=t["ink3"],
                linewidth=4, solid_capstyle="butt", zorder=2)
    ax.text(duration * 1.005, 0.86, f"{len(events)} tunnels", fontsize=10.5,
            color=t["ink3"], va="center")

    def row(times, y, colour, name, emphasise):
        """Draw where one strategy looked, and count what it actually hit.

        The counts are computed here rather than written into the label: a
        banner that says "0 found" next to a visible hit is the fastest way to
        lose a reader, and hard-coding invites exactly that.
        """
        ax.plot([0, duration], [y, y], color=t["grid"], linewidth=1, zorder=1)
        on = [x for x in times
              if any(a - 2 <= x <= b + 2 for a, b, _ in events)]
        off = [x for x in times if x not in on]
        ax.scatter(off, [y] * len(off), s=17, color=colour, zorder=3,
                   linewidths=0)
        if on:
            ax.scatter(on, [y] * len(on), s=95, marker="*", color=colour,
                       zorder=4, linewidths=0)
        # distinct tunnels reached, which is what a user cares about -- two
        # calls landing in the same tunnel is one tunnel found
        hit_events = sum(1 for a, b, _ in events
                         if any(a - 2 <= x <= b + 2 for x in times))
        ax.text(duration * 1.005, y, f"{name}: {hit_events} of {len(events)}",
                fontsize=10.5, color=colour, va="center",
                fontweight="600" if emphasise else "normal")
        return hit_events

    n_uniform = row(picks["uniform"], 0.44, t["muted"], "uniform", False)
    n_ours = row(picks.get("segment_adaptive", picks.get("segment", [])), 0.06,
                 t["series"][0], "framesieve", True)

    fig.text(0.035, 0.012,
             f"Every grey bar is a tunnel the ground truth marks. Both rows spent "
             f"the same 32 model calls. Over 200 random phases uniform finds "
             f"nothing at all {100*hero['uniform_miss_rate']:.0f}% of the time.",
             fontsize=10, color=t["ink3"])
    print(f"  [{mode}] uniform hit {n_uniform} tunnels, framesieve {n_ours}")
    return save(fig, os.path.join(ROOT, "figures/banner.png"), mode)


def at_a_glance(mode: str = "light") -> str:
    """Three measured numbers, as a strip, with what each is against."""
    t = THEME[mode]
    cards = [
        ("15 s", "to index an hour of video", "and 5 MB, measured over 205 h"),
        ("25 ms", "to search all 4.5 hours", "measured on 20 fresh queries"),
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
        print(at_a_glance(mode))
