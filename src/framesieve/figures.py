"""Figures for the post.

House style, applied everywhere:
  - marks carry the data, everything else recedes: no top/right spine, a single
    hairline y-grid, axis text in ink rather than series colour
  - four series maximum, direct-labelled at the line end, because two of the
    light-mode hues sit below 3:1 against the surface and colour alone is not
    allowed to carry identity
  - log x wherever the x axis is a compute budget, because the interesting
    behaviour is at the cheap end and a linear axis hides it
  - every figure renders in both light and dark, from the same validated steps

Palette is the dataviz reference instance, checked with bench/validate_palette.py
(worst adjacent CVD dE 9.1 light / 8.4 dark; both modes PASS).
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

THEME = {
    "light": dict(
        surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", ink3="#8a8880",
        grid="#e4e3de",
        series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"],
        seq="#2a78d6", muted="#b9b7ae"),
    "dark": dict(
        surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", ink3="#8a8880",
        grid="#33322f",
        series=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"],
        seq="#3987e5", muted="#575651"),
}

# Strategy -> fixed slot, so a figure that drops a series never repaints the
# rest. Five series is at the soft cap; it stays gate-safe here because line
# charts use the *adjacent* pairlist (validated: worst adjacent CVD dE 9.1 light
# / 8.4 dark) and every series is direct-labelled, which the light-mode contrast
# relief requires anyway.
STRATEGY_SLOT = {"uniform": 0, "topk": 1, "nms": 2, "segment": 3,
                 "segment_adaptive": 4}
STRATEGY_LABEL = {"uniform": "uniform", "topk": "top-k",
                  "nms": "top-k + NMS", "segment": "segment (fixed τ)",
                  "segment_adaptive": "segment (budget-adaptive)"}


def style(mode: str = "light"):
    t = THEME[mode]
    plt.rcParams.update({
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"], "text.color": t["ink"],
        "axes.labelcolor": t["ink2"], "xtick.color": t["ink2"],
        "ytick.color": t["ink2"], "axes.edgecolor": t["grid"],
        "grid.color": t["grid"], "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10.5, "axes.titlesize": 12.5, "axes.titleweight": "600",
        "figure.dpi": 160, "lines.linewidth": 2.0, "lines.markersize": 5.5,
        "legend.frameon": False,
    })
    return t


def label_line_ends(ax, ends: Sequence[tuple], x_at: float, t,
                    min_gap: float = 0.058, x_pad: float = 1.16):
    """Direct-label a set of lines at their right ends without letting the labels
    stack on top of each other.

    Labels are pushed apart vertically and keep a leader line back to their own
    line end, so nudging one never makes it ambiguous which series it belongs to.
    Direct labels are not optional here: several of the light-mode hues sit below
    3:1 against the surface, which the palette's contrast relief requires them for.
    """
    ends = sorted(ends, key=lambda e: -e[0])
    placed: list[float] = []
    for val, _, _ in ends:
        placed.append(val if not placed else min(val, placed[-1] - min_gap))
    for (val, label, c), yy in zip(ends, placed):
        ax.annotate(label, xy=(x_at, val), xytext=(x_at * x_pad, yy), color=c,
                    fontsize=9.5, fontweight="600", va="center",
                    annotation_clip=False,
                    arrowprops=(dict(arrowstyle="-", color=c, linewidth=0.8,
                                     shrinkA=0, shrinkB=2, alpha=0.6)
                                if abs(yy - val) > 0.012 else None))


def _finish(ax, t, title: str, sub: str = "", xlabel: str = "", ylabel: str = ""):
    ax.set_title(title, color=t["ink"], loc="left", pad=16 if sub else 8)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, color=t["ink2"],
                fontsize=9.5, va="bottom")
    ax.set_xlabel(xlabel, fontsize=9.5)
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.grid(axis="y", alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def save(fig, path: str, mode: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    stem, ext = os.path.splitext(path)
    out = f"{stem}.{mode}{ext or '.png'}"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.28)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------


def fig_recall_curve(rows: list[dict], mode: str = "light",
                     metric: str = "event_recall",
                     out: str = "figures/recall_curve.png",
                     title: str = "What the cascade finds, per VLM call",
                     sub: str = "") -> str:
    """The headline: recall against the compute you spend to get it."""
    t = style(mode)
    fig, ax = plt.subplots(figsize=(7.4, 4.6))

    strategies = [s for s in STRATEGY_SLOT if any(r["strategy"] == s for r in rows)]
    ends: list[tuple[float, str, str]] = []
    for s in strategies:
        rs = sorted([r for r in rows if r["strategy"] == s], key=lambda r: r["budget"])
        x = np.array([r["budget"] for r in rs], float)
        y = np.array([r[metric] for r in rs], float)
        c = t["series"][STRATEGY_SLOT[s]]
        lo = np.array([r.get(metric + "_lo", np.nan) for r in rs], float)
        hi = np.array([r.get(metric + "_hi", np.nan) for r in rs], float)
        if np.isfinite(lo).all() and np.isfinite(hi).all():
            # five overlapping bands turn the panel to mud; keep them faint
            # enough to read the lines through and dense enough to still say
            # "these differences are inside the noise"
            ax.fill_between(x, lo, hi, color=c, alpha=0.07, linewidth=0)
        ax.plot(x, y, color=c, marker="o", zorder=3,
                markeredgecolor=t["surface"], markeredgewidth=1.2)
        ends.append((float(y[-1]), STRATEGY_LABEL[s], c))

    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 1.02)
    xmax = max(r["budget"] for r in rows)
    ax.set_xlim(min(r["budget"] for r in rows) * 0.85, xmax * 1.15)

    label_line_ends(ax, ends, xmax, t)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    _finish(ax, t, title, sub, "VLM calls per query  (log)",
            "event recall" if metric == "event_recall" else metric.replace("_", " "))
    return save(fig, out, mode)


def fig_cost_hierarchy(stages: list[tuple[str, float]], mode: str = "light",
                       out: str = "figures/cost_hierarchy.png",
                       title: str = "Cost of looking at one frame",
                       sub: str = "") -> str:
    """Magnitude comparison, so: one hue, more-is-darker, horizontal bars."""
    t = style(mode)
    fig, ax = plt.subplots(figsize=(7.4, 0.52 * len(stages) + 1.9))
    names = [s[0] for s in stages]
    vals = np.array([s[1] for s in stages], float)
    order = np.argsort(vals)
    names = [names[i] for i in order]
    vals = vals[order]

    # sequential: one hue, light -> dark with magnitude. Steps are a straight
    # mix toward the surface so the lightest bar still reads as the same hue.
    from matplotlib.colors import to_rgb
    base = np.array(to_rgb(t["seq"]))
    surf = np.array(to_rgb(t["surface"]))
    shades = [tuple(np.clip(surf + (base - surf) * w, 0, 1))
              for w in np.linspace(0.34, 1.0, len(vals))]

    y = np.arange(len(vals))
    ax.barh(y, vals, color=shades, height=0.62)
    ax.set_yticks(y, names, fontsize=9.5)
    ax.set_xscale("log")
    for i, v in enumerate(vals):
        lab = f"{v*1000:.2f} ms" if v < 1 else f"{v:.2f} s"
        ax.text(v * 1.12, i, lab, va="center", color=t["ink2"], fontsize=9)
    ax.set_xlim(vals.min() * 0.5, vals.max() * 6)
    _finish(ax, t, title, sub, "seconds per frame  (log)", "")
    ax.grid(axis="x", alpha=0.9)
    ax.grid(axis="y", visible=False)
    return save(fig, out, mode)


def fig_decode_scaling(rows: list[dict], mode: str = "light",
                       out: str = "figures/decode_scaling.png",
                       title: str = "Decode is not the bottleneck",
                       sub: str = "") -> str:
    t = style(mode)
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for k, backend in enumerate(["cpu", "nvdec"]):
        rs = sorted([r for r in rows if r["backend"] == backend],
                    key=lambda r: r["pixels"])
        if not rs:
            continue
        x = np.array([r["pixels"] / 1e6 for r in rs])
        y = np.array([r["realtime_factor"] for r in rs])
        c = t["series"][k]
        ax.plot(x, y, color=c, marker="o", zorder=3,
                markeredgecolor=t["surface"], markeredgewidth=1.2)
        ax.annotate(backend.upper() if backend == "cpu" else "NVDEC",
                    (x[-1], y[-1]), color=c, fontsize=9.5, fontweight="600",
                    xytext=(7, 0), textcoords="offset points", va="center")
    ax.axhline(1.0, color=t["muted"], linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    # axes-fraction x, data y: reading xlim on a log axis can return ~0, and a
    # tight bounding box around a point at log(0) grows the canvas without limit
    ax.text(0.01, 1.0, " realtime", color=t["ink3"], fontsize=9, va="bottom",
            transform=ax.get_yaxis_transform())
    ax.set_xscale("log"); ax.set_yscale("log")
    _finish(ax, t, title, sub, "megapixels per frame  (log)",
            "x realtime  (log)")
    return save(fig, out, mode)


def fig_hero_timeline(duration_s: float, events: Sequence[tuple[float, float]],
                      picks: dict[str, np.ndarray], mode: str = "light",
                      out: str = "figures/hero_timeline.png",
                      title: str = "", sub: str = "") -> str:
    """Where the event is, and where each strategy chose to look.

    Emphasis rather than categorical: the event is the subject, the samplers are
    context, so the event band is the only saturated thing on the page.
    """
    t = style(mode)
    n = len(picks)
    fig, ax = plt.subplots(figsize=(8.2, 0.62 * n + 1.9))

    # The events are context, not a series, so they are drawn achromatically: a
    # fifth hue fails the all-pairs colour gates against the four strategy hues,
    # and it would read as "another method" besides.
    #
    # A three-second event inside a four-hour video is far narrower than a pixel,
    # so each one needs a minimum drawn width -- but sixty of them at a generous
    # minimum width cover a third of the axis and the figure becomes a barcode.
    # So the *total* event ink is budgeted instead: every event is widened
    # equally, up to a combined 10% of the axis.
    n_ev = max(1, len(events))
    min_w = min(duration_s * 0.005, duration_s * 0.10 / n_ev)
    for a, b in events:
        w = max(b - a, min_w)
        mid = (a + b) / 2
        ax.axvspan(mid - w / 2, mid + w / 2, color=t["ink"], alpha=0.5,
                   zorder=2, linewidth=0)
        if n_ev <= 8:      # edge lines only help when there are few enough to see
            for edge in (mid - w / 2, mid + w / 2):
                ax.axvline(edge, color=t["ink2"], linewidth=1.2, zorder=2)

    for j, (name, ts) in enumerate(picks.items()):
        y = n - 1 - j
        ax.axhline(y, color=t["grid"], linewidth=1.0, zorder=1)
        hit = np.zeros(len(ts), bool)
        for a, b in events:
            hit |= (ts >= a - 0.5) & (ts <= b + 0.5)
        c = t["series"][STRATEGY_SLOT.get(name, 0)]
        ax.scatter(ts[~hit], np.full((~hit).sum(), y), s=26, color=c, alpha=0.55,
                   linewidths=0, zorder=3)
        if hit.any():
            ax.scatter(ts[hit], np.full(hit.sum(), y), s=150, color=c,
                       edgecolor=t["surface"], linewidths=1.6, zorder=6, marker="*")
        # identity is carried by the coloured marks on the row; the label stays
        # in ink, and the outcome is spelled out rather than encoded in colour
        ax.text(duration_s * 1.02, y, STRATEGY_LABEL.get(name, name),
                va="center", fontsize=9.5, color=t["ink"], fontweight="600")
        ax.text(duration_s * 1.02, y - 0.26,
                "found it" if hit.any() else "missed it entirely",
                va="center", fontsize=8.5, color=t["ink2"] if hit.any() else t["ink"],
                fontweight="600" if not hit.any() else "normal")
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlim(0, duration_s)
    ax.set_yticks([])
    ax.set_xticks(np.linspace(0, duration_s, 7),
                  [f"{int(v//3600)}:{int(v%3600//60):02d}"
                   for v in np.linspace(0, duration_s, 7)])
    _finish(ax, t, title, sub, "position in video  (h:mm)", "")
    ax.grid(visible=False)
    ax.spines["left"].set_visible(False)
    return save(fig, out, mode)


def fig_redundancy(ratios: Sequence[float], mode: str = "light",
                   out: str = "figures/redundancy.png",
                   title: str = "How much of a video is the same shot again",
                   sub: str = "") -> str:
    """One variable's distribution, so: one hue, and a log x because the tail is
    the whole story -- a linear axis would compress 95% of the mass into one bar."""
    t = style(mode)
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    r = np.asarray(ratios, float)
    bins = np.logspace(np.log10(r.min() * 0.95), np.log10(r.max() * 1.05), 34)
    ax.hist(r, bins=bins, color=t["seq"], edgecolor=t["surface"], linewidth=0.8)
    med = float(np.median(r))
    ax.axvline(med, color=t["ink2"], linewidth=1.4, linestyle=(0, (4, 3)), zorder=4)
    ax.annotate(f"median {med:.1f}×", (med, ax.get_ylim()[1] * 0.92),
                color=t["ink"], fontsize=9.5, fontweight="600",
                xytext=(8, 0), textcoords="offset points", va="top")
    p95 = float(np.percentile(r, 95))
    ax.annotate(f"p95 {p95:.0f}×", (p95, ax.get_ylim()[1] * 0.55),
                color=t["ink2"], fontsize=9.5, xytext=(8, 0),
                textcoords="offset points", va="top")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:g}×")
    _finish(ax, t, title, sub, "frames per distinct segment  (log)", "videos")
    return save(fig, out, mode)


def fig_tau_sweep(entries: list[dict], mode: str = "light",
                  out: str = "figures/tau_sweep.png",
                  strategy: str = "segment",
                  title: str = "How fine should the segments be?",
                  sub: str = "") -> str:
    """Recall vs budget, one line per segment_tau.

    tau is an *ordered* variable, so it gets a sequential ramp (one hue, coarse to
    fine) rather than categorical hues -- categorical would imply the taus are
    unrelated categories and would make the trend across them unreadable. The
    tau=0 control, where `segment` is exactly `topk`, is drawn as the dashed
    reference it is.
    """
    t = style(mode)
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    from matplotlib.colors import to_rgb

    base = np.array(to_rgb(t["seq"]))
    surf = np.array(to_rgb(t["surface"]))
    ent = sorted(entries, key=lambda e: e["segment_tau"])
    steps = np.linspace(0.38, 1.0, max(1, len(ent)))
    ends: list[tuple[float, str, str]] = []

    for k, e in enumerate(ent):
        rs = sorted([r for r in e["rows"] if r["strategy"] == strategy],
                    key=lambda r: r["budget"])
        if not rs:
            continue
        x = [r["budget"] for r in rs]
        y = [r["event_recall"] for r in rs]
        tau = e["segment_tau"]
        control = tau == 0
        c = t["ink3"] if control else tuple(np.clip(surf + (base - surf) * steps[k], 0, 1))
        ax.plot(x, y, color=c, marker="o", zorder=3,
                linestyle=(0, (4, 3)) if control else "-",
                markeredgecolor=t["surface"], markeredgewidth=1.2)
        lab = ("no collapse (= top-k)" if control
               else f"τ={tau:g}  ({e['n_segments']:,} seg)")
        ends.append((float(y[-1]), lab, t["ink2"] if control else c))

    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    xs = [r["budget"] for e in ent for r in e["rows"]]
    xmax = max(xs)
    ax.set_xlim(min(xs) * 0.85, xmax * 1.15)
    label_line_ends(ax, ends, xmax, t)
    _finish(ax, t, title, sub, "VLM calls per query  (log)", "event recall")
    return save(fig, out, mode)


def fig_pareto(points: list[dict], mode: str = "light",
               out: str = "figures/pareto.png", title: str = "",
               sub: str = "", xkey: str = "vlm_gpu_s", ykey: str = "accuracy",
               xlabel: str = "GPU-seconds per query  (log)",
               ylabel: str = "accuracy") -> str:
    """Accuracy against compute, with prior work drawn as reference marks."""
    t = style(mode)
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    strategies = [s for s in STRATEGY_SLOT
                  if any(p.get("strategy") == s for p in points)]
    ends: list[tuple] = []
    for s in strategies:
        ps = sorted([p for p in points if p.get("strategy") == s], key=lambda p: p[xkey])
        c = t["series"][STRATEGY_SLOT[s]]
        ax.plot([p[xkey] for p in ps], [p[ykey] for p in ps], color=c, marker="o",
                zorder=3, markeredgecolor=t["surface"], markeredgewidth=1.2)
        # label each point with the budget it cost, so the x axis reads as a
        # decision ("how many frames?") and not just an abstract compute number
        if s == strategies[0]:
            for p in ps:
                if "budget" in p:
                    ax.annotate(f"K={p['budget']}", (p[xkey], p[ykey]), color=t["ink3"],
                                fontsize=8.5, xytext=(-2, -15), ha="right",
                                textcoords="offset points")
        ends.append((float(ps[-1][ykey]), STRATEGY_LABEL[s], c))
    for p in [p for p in points if p.get("kind") == "reference"]:
        ax.axhline(p[ykey], color=t["muted"], linewidth=1.2,
                   linestyle=(0, (4, 3)), zorder=1)
        ax.text(0.01, p[ykey], f" {p['label']}", color=t["ink3"], fontsize=9,
                va="bottom", transform=ax.get_yaxis_transform())
    ax.set_xscale("log")
    xs = [p[xkey] for p in points if np.isfinite(p.get(xkey, np.nan))]
    if xs:
        ax.set_xlim(min(xs) * 0.8, max(xs) * 1.15)
        label_line_ends(ax, ends, max(xs), t, min_gap=0.012, x_pad=1.10)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    _finish(ax, t, title, sub, xlabel, ylabel)
    return save(fig, out, mode)


def fig_ceiling(ceiling: dict, measured: dict, mode: str = "light",
                out: str = "figures/ceiling.png",
                title: str = "How good could selection possibly be?",
                sub: str = "") -> str:
    """The ceiling top-k can reach, what it actually reaches, and what diversity
    adds on top.

    The ceiling is drawn as a reference line rather than a series -- it is not a
    method, it is the limit the ranking imposes. The shaded band between top-k
    and the best strategy is the only thing selection cleverness buys, which is
    the point of the figure.
    """
    t = style(mode)
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ks = sorted(int(k) for k in ceiling)
    ceil = np.array([ceiling[str(k)] for k in ks], float)
    tk = np.array([measured["topk"][str(k)] for k in ks], float)
    best = np.array([max(measured[s][str(k)] for s in measured) for k in ks], float)

    ax.fill_between(ks, tk, best, color=t["series"][4], alpha=0.16, linewidth=0,
                    zorder=2)
    ax.plot(ks, ceil, color=t["ink2"], linewidth=1.6, linestyle=(0, (5, 3)),
            zorder=3)
    ax.plot(ks, tk, color=t["series"][1], marker="o", zorder=4,
            markeredgecolor=t["surface"], markeredgewidth=1.2)
    ax.plot(ks, best, color=t["series"][4], marker="o", zorder=4,
            markeredgecolor=t["surface"], markeredgewidth=1.2)

    ends = [(float(best[-1]), "best selector", t["series"][4]),
            (float(tk[-1]), "top-k", t["series"][1]),
            (float(ceil[-1]), "top-k ceiling", t["ink2"])]
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(min(ks) * 0.85, max(ks) * 1.15)
    label_line_ends(ax, ends, max(ks), t, min_gap=0.05)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    _finish(ax, t, title, sub, "VLM calls per query  (log)", "event recall")
    return save(fig, out, mode)


def fig_confidence(strata: list[dict], mode: str = "light",
                   out: str = "figures/confidence.png",
                   key: str = "segment_adaptive",
                   title: str = "Most of what is “missed” was barely there",
                   sub: str = "") -> str:
    """Recall against budget, one line per floor on the oracle's confidence.

    The floor is an ordered variable, so it gets a sequential ramp rather than
    categorical hues: the reader's question is "which way does it move", not
    "which category is this".
    """
    t = style(mode)
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    from matplotlib.colors import to_rgb
    base = np.array(to_rgb(t["seq"]))
    surf = np.array(to_rgb(t["surface"]))
    steps = np.linspace(0.34, 1.0, max(1, len(strata)))
    ends = []
    for i, s in enumerate(strata):
        ks = sorted(int(k.split("_")[-1]) for k in s if k.startswith(key))
        y = [s[f"{key}_{k}"] for k in ks]
        c = tuple(np.clip(surf + (base - surf) * steps[i], 0, 1))
        ax.plot(ks, y, color=c, marker="o", zorder=3,
                markeredgecolor=t["surface"], markeredgewidth=1.2)
        ends.append((float(y[-1]), f"{s['label']}  ({s['n_events']} events)", c))
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(min(ks) * 0.85, max(ks) * 1.15)
    label_line_ends(ax, ends, max(ks), t, min_gap=0.055)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    _finish(ax, t, title, sub, "VLM calls per query  (log)", "event recall")
    return save(fig, out, mode)


def fig_oracle_examples(pairs: list[dict], mode: str = "light",
                        out: str = "figures/oracle_examples.png",
                        title: str = "What a low-confidence label looks like",
                        sub: str = "") -> str:
    """Two frames the oracle labelled positive for the same query, at opposite
    ends of its confidence.

    The confidence-stratified recall curve is an argument from correlation; this
    is the argument from looking. Frames are shown as captured, with the oracle's
    own score, so the reader can judge whether a retriever missing the left-hand
    one is a failure.
    """
    import matplotlib.image as mpimg

    t = style(mode)
    fig, axes = plt.subplots(1, len(pairs), figsize=(4.9 * len(pairs), 4.05))
    if len(pairs) == 1:
        axes = [axes]
    for ax, p in zip(axes, pairs):
        ax.imshow(mpimg.imread(p["path"]))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(t["grid"]); s.set_linewidth(1)
        ax.set_title(p["label"], color=t["ink"], fontsize=11, loc="left",
                     fontweight="600", pad=8)
        ax.text(0.0, -0.035, p["note"], transform=ax.transAxes, color=t["ink2"],
                fontsize=9.5, va="top", ha="left")
    # place both header lines explicitly; a suptitle plus fig.text at nearby
    # y values collide as soon as the figure height changes
    fig.subplots_adjust(top=0.80, bottom=0.10, wspace=0.06)
    fig.text(0.005, 0.985, title, ha="left", va="top", color=t["ink"],
             fontsize=12.5, fontweight="600")
    if sub:
        fig.text(0.005, 0.925, sub, ha="left", va="top", color=t["ink2"],
                 fontsize=9.5)
    return save(fig, out, mode)


def fig_interior(fams: list[dict], mean_r1: float, max_r1: float,
                 mode: str = "light", path: str | None = None):
    """Four unrelated ways of interpolating between the mean and the max.

    Small multiples rather than one panel: the families have genuinely different
    x-axes (a count, an exponent, a divisor, a weight) and forcing them onto a
    shared "how max-like is this" axis would be inventing a quantity none of them
    measures. What has to be comparable is the *y* axis and the two reference
    lines, so those are shared and drawn identically in every panel.
    """
    t = style(mode)
    fig, axes = plt.subplots(1, len(fams), figsize=(4.05 * len(fams), 3.5),
                             sharey=True)
    axes = np.atleast_1d(axes)
    lo = min(min(f["y"]) for f in fams) - 0.6
    hi = max(max(f["y"]) for f in fams) + 1.1

    for i, (ax, f) in enumerate(zip(axes, fams)):
        c = t["series"][i]
        for ref, lab in ((mean_r1, "plain mean"), (max_r1, "plain max")):
            ax.axhline(ref, color=t["muted"], linewidth=1.1, linestyle=(0, (4, 3)),
                       zorder=1)
            # named once, on the last panel, where nothing else is competing
            # for the space -- repeating them four times is noise
            if i == len(fams) - 1:
                ax.text(0.985, ref + 0.1, lab, transform=ax.get_yaxis_transform(),
                        color=t["ink3"], fontsize=8.5, va="bottom", ha="right")
        ax.plot(range(len(f["x"])), f["y"], color=c, marker="o",
                markerfacecolor=t["surface"], markeredgecolor=c,
                markeredgewidth=1.8, zorder=3)
        j = int(np.argmax(f["y"]))
        ax.plot([j], [f["y"][j]], marker="o", markersize=8.5, color=c, zorder=4)
        ax.annotate(f"{f['y'][j]:.2f}", xy=(j, f["y"][j]),
                    xytext=(0, 9), textcoords="offset points",
                    ha="center", color=c, fontsize=9.5, fontweight="600")
        ax.set_xticks(range(len(f["x"])))
        ax.set_xticklabels(f["x"], fontsize=8.5)
        ax.set_xlim(-0.5, len(f["x"]) - 0.5)
        ax.set_ylim(lo, hi)
        _finish(ax, t, f["title"], f.get("sub", ""), f.get("xlabel", ""),
                "R@1" if i == 0 else "")
        ax.grid(axis="y", alpha=0.9)
        # the endpoints of each family, named where they actually sit
        for side, name in ((0, f.get("left", "")), (len(f["x"]) - 1, f.get("right", ""))):
            if name:
                ax.text(side, lo + 0.16, name, ha="center", color=t["ink3"],
                        fontsize=8.5)
    fig.suptitle("")
    fig.tight_layout()
    return save(fig, path or "figures/interior.png", mode)


def fig_routing(quartiles: list[dict], mode: str = "light",
                path: str | None = None):
    """Where in the ranking the answer sits, by confidence quartile.

    A stacked bar is the right form here because the five bands are parts of one
    whole (every query is in exactly one), and the argument is about how that
    whole is composed rather than about any single band's size. The two bands
    where an extra call cannot help are drawn in the muted neutral, so the
    cancellation reads without needing the numbers.
    """
    t = style(mode)
    fig, ax = plt.subplots(figsize=(8.6, 3.5))
    bands = [("already rank 0", t["muted"], "no call helps"),
             ("ranks 1-3", t["series"][0], ""),
             ("ranks 4-9", t["series"][2], ""),
             ("ranks 10-31", t["series"][3], ""),
             ("deeper or absent", t["muted"], "no call helps")]
    ys = np.arange(len(quartiles))[::-1]
    left = np.zeros(len(quartiles))
    for bi, (lab, c, _) in enumerate(bands):
        w = np.array([q["bands"][bi] for q in quartiles], float)
        ax.barh(ys, w, left=left, height=0.62, color=c,
                edgecolor=t["surface"], linewidth=2.0, zorder=3)
        for y, x0, ww in zip(ys, left, w):
            if ww >= 5.5:
                ax.text(x0 + ww / 2, y, f"{ww:.0f}", ha="center", va="center",
                        color=t["surface"] if c != t["muted"] else t["ink"],
                        fontsize=9, fontweight="600", zorder=4)
        left = left + w
    ax.set_yticks(ys)
    ax.set_yticklabels([q["label"] for q in quartiles], fontsize=9.5)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100%"])
    _finish(ax, t, "Where the answer sits, by retrieval confidence",
            "the coloured span is the only place another call can pay off",
            "share of queries", "")
    ax.grid(axis="x", alpha=0.9)
    ax.grid(axis="y", visible=False)
    # the two grey bands mean the same thing -- a call that cannot change the
    # answer -- so they share one legend entry rather than pretending to differ
    keys = [(t["muted"], "no call helps"), (bands[1][1], bands[1][0]),
            (bands[2][1], bands[2][0]), (bands[3][1], bands[3][0])]
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c, _ in keys],
              [lab for _, lab in keys], loc="lower center",
              bbox_to_anchor=(0.5, -0.42), ncol=4, fontsize=9)
    fig.tight_layout()
    return save(fig, path or "figures/routing.png", mode)


def fig_k_star(synth: dict, real: dict, anchors: list, mode: str = "light",
               path: str | None = None):
    """The whole thesis in one chart: the best k tracks m.

    Log-log with the identity line drawn, because the claim is proportionality
    and a linear axis would make the small-m end -- where max lives, and where
    most people are -- unreadable. Two measured series plus the two real systems
    that sit at the endpoints, so the reader can see that max and mean are not
    wrong, they are right in exactly one place each.
    """
    t = style(mode)
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    lim = (0.8, 40)
    ax.plot(lim, lim, color=t["ink3"], linewidth=1.1, linestyle=(0, (4, 3)),
            zorder=1)
    ax.annotate("k = m", xy=(26, 26), xytext=(30, 20), color=t["ink3"],
                fontsize=9.5, ha="left", va="center")

    for i, (lab, d) in enumerate((("synthetic, m known by construction", synth),
                                  ("real video, m measured by a dense VLM", real))):
        c = t["series"][i]
        ax.plot(d["m"], d["k"], marker="o", color=c, linewidth=2.0,
                markerfacecolor=t["surface"], markeredgecolor=c,
                markeredgewidth=1.8, zorder=3, label=lab)
    # Real systems that sit at an endpoint are annotated in place. They land on
    # top of the curves' own first point, so they are drawn as a ring around it
    # with the label pushed clear of the axis rather than under it.
    for m, k, name, dy in anchors:
        ax.plot([m], [k], marker="o", markersize=13, markerfacecolor="none",
                markeredgecolor=t["series"][3], markeredgewidth=2.0, zorder=4)
        ax.annotate(name, xy=(m, k), xytext=(22, dy), textcoords="offset points",
                    ha="left", va="bottom", color=t["series"][3],
                    fontsize=9.5, fontweight="600",
                    arrowprops=dict(arrowstyle="-", color=t["series"][3],
                                    linewidth=0.9, shrinkA=0, shrinkB=9,
                                    alpha=0.7))

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    for v in (1, 2, 4, 8, 16, 32):
        pass
    ax.set_xticks([1, 2, 4, 8, 16, 32]); ax.set_yticks([1, 2, 4, 8, 16, 32])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32"])
    ax.set_yticklabels(["1\n(max)", "2", "4", "8", "16", "32"])
    _finish(ax, t, "The best pooling depth is a property of your data",
            "how many of the pooled items genuinely match, against the k that scores best",
            "m  \u2014  items that genuinely match", "best k")
    ax.set_title(ax.get_title(loc="left"), color=t["ink"], loc="left", pad=26)
    ax.grid(axis="x", alpha=0.6)
    ax.legend(loc="upper left", fontsize=9.5)
    fig.tight_layout()
    return save(fig, path or "figures/k_star.png", mode)


def fig_endpoint_cost(rows: list, mode: str = "light",
                      path: str | None = None):
    """What each endpoint costs against a tuned k, as a function of m.

    Two lines rather than a grouped bar: the message is that the two curves
    cross, so each endpoint owns one side of the range and neither is a default.
    """
    t = style(mode)
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ms = [r["m"] for r in rows]
    lost_max = [100 * (r["best"] - r["max"]) for r in rows]
    lost_mean = [100 * (r["best"] - r["mean"]) for r in rows]
    for y, lab, i in ((lost_max, "cost of using max", 1),
                      (lost_mean, "cost of using mean", 2)):
        c = t["series"][i]
        ax.plot(ms, y, marker="o", color=c, linewidth=2.0,
                markerfacecolor=t["surface"], markeredgecolor=c,
                markeredgewidth=1.8, zorder=3)
    label_line_ends(ax, [(lost_max[-1], "max", t["series"][1]),
                         (lost_mean[-1], "mean", t["series"][2])],
                    ms[-1], t, min_gap=4.0, x_pad=1.06)
    ax.set_xscale("log")
    ax.set_xticks(ms); ax.set_xticklabels([str(m) for m in ms])
    ax.set_xlim(min(ms) * 0.9, max(ms) * 1.35)
    ax.axhline(0, color=t["ink3"], linewidth=1.0)
    _finish(ax, t, "Each endpoint is right in exactly one place",
            "R@1 given up by using an endpoint instead of the best k, at that m",
            "m  —  items that genuinely match", "points of R@1 lost")
    fig.tight_layout()
    return save(fig, path or "figures/endpoint_cost.png", mode)


def fig_metric_curves(curves: list, mode: str = "light",
                      path: str | None = None):
    """Where each metric puts the pooling optimum.

    Each curve is normalised to its OWN range over k, because the alternative --
    comparing gain magnitudes across metrics -- is confounded: R@1, nDCG and AUC
    have different scales and different ceilings, so "this metric shows a bigger
    improvement" is not a statement those numbers can support. Normalised within
    a metric, the only thing being compared is WHERE the peak sits, which is a
    fair comparison and is the claim.
    """
    t = style(mode)
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    ends = []
    for i, c in enumerate(curves):
        col = t["series"][i]
        y = np.asarray(c["y"], dtype=float)
        y = (y - y.min()) / (y.max() - y.min() + 1e-12)
        ax.plot(c["ks"], y, color=col, linewidth=2.0, marker="o",
                markerfacecolor=t["surface"], markeredgecolor=col,
                markeredgewidth=1.6, zorder=3)
        j = int(np.argmax(y))
        ax.plot([c["ks"][j]], [y[j]], marker="o", markersize=9, color=col, zorder=4)
        ax.annotate(f"k={c['ks'][j]}", xy=(c["ks"][j], y[j]),
                    xytext=(0, 11), textcoords="offset points", ha="center",
                    color=col, fontsize=9.5, fontweight="600")
        ends.append((float(y[-1]), c["label"], col))
    ax.set_xscale("log")
    ax.set_xticks(curves[0]["ks"])
    ax.set_xticklabels([str(k) for k in curves[0]["ks"]])
    ax.set_ylim(-0.06, 1.16)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["worst k", "", "best k"])
    label_line_ends(ax, ends, curves[0]["ks"][-1], t, min_gap=0.1, x_pad=1.1)
    _finish(ax, t, "Which k your metric will tell you to use",
            "one fixed set of scores; each curve normalised to its own range",
            "k  \u2014  pooling depth", "")
    fig.tight_layout()
    return save(fig, path or "figures/metric_curves.png", mode)


def fig_metric_gain(rows: list, mode: str = "light", path: str | None = None):
    """How much leaving `max` is worth, by how deep the metric looks.

    A horizontal bar on a shared axis, because the comparison is one quantity
    across seven labelled cases and the labels are words, not numbers. Ordered by
    metric depth rather than by value, so the monotone trend is the shape of the
    chart rather than something the reader has to reconstruct.
    """
    t = style(mode)
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    labs = [r["metric"] for r in rows]
    vals = [r["rel_gain_pct"] for r in rows]
    ys = np.arange(len(rows))[::-1]
    # one accent for the head metrics, the muted neutral for the deep ones, so
    # the divide the post is about is visible without reading the numbers
    cols = [t["series"][1] if r["depth"] <= 5 else t["muted"] for r in rows]
    ax.barh(ys, vals, height=0.62, color=cols, edgecolor=t["surface"],
            linewidth=2.0, zorder=3)
    for y, v, c in zip(ys, vals, cols):
        ax.text(v + max(vals) * 0.015, y, f"{v:.0f}%", va="center", ha="left",
                color=c if c != t["muted"] else t["ink2"],
                fontsize=10, fontweight="600", zorder=4)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{name}   " + (f"top {r['depth']}" if r["depth"] <= 20
                                        else "whole ranking")
                        for name, r in zip(labs, rows)], fontsize=9.5)
    ax.set_xlim(0, max(vals) * 1.16)
    _finish(ax, t, "The same improvement, priced by seven metrics",
            "relative gain from moving off max, on one fixed set of scores",
            "gain over max (%)", "")
    ax.grid(axis="x", alpha=0.9)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return save(fig, path or "figures/metric_gain.png", mode)
