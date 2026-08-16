"""Assemble the publishable posts: inline the figures, substitute measured numbers.

Two pages now, sharing one stylesheet and one substitution pass:

    docs/post.html      framesieve -- the video search tool this repo is
    docs/pooling.html   the score-pooling finding that fell out of building it

They are built together rather than by two drifting copies of this file. The
Artifact CSP blocks every external host, so each page carries its own images;
both theme variants are embedded and swapped in CSS. Numbers are pulled from the
run artifacts rather than typed, so the prose cannot drift from the measurements.
"""
from __future__ import annotations

import base64
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGES = [("docs/post.template.html", "docs/post.html"),
         ("docs/pooling.template.html", "docs/pooling.html")]
FIGS = os.path.join(ROOT, "figures")

TIMED = {"t0": "runs/ms_t0.json", "t1": "runs/ms_t1.json", "t4": "runs/ms_t4.json",
         "msf_topk": "runs/msf_topk.json", "msf_topk_c5": "runs/msf_topk_c5.json",
         "msf_topk_c10": "runs/msf_topk_c10.json"}


def load_timing(key: str):
    p = os.path.join(ROOT, TIMED[key])
    if not os.path.exists(p):
        return None
    row = json.load(open(p))["rows"][0]
    n = max(1, row["n_queries"])
    return {k: row.get(k, 0.0) / n * 1000 for k in ("select_s", "fetch_s", "vlm_s")}


def fmt_ms(v: float) -> str:
    return f"{v:.1f} ms" if v < 10 else (f"{v:.0f} ms" if v < 1000 else f"{v/1000:.2f} s")


times = {k: load_timing(k) for k in TIMED}
absent = [k for k, v in times.items() if v is None]
if absent:
    sys.exit(f"missing timing runs: {absent} -- run scripts/run_ms_timed.sh")


def total(k: str) -> float:
    t = times[k]
    return t["select_s"] + t["fetch_s"] + t["vlm_s"]


def latency_note() -> str:
    t = times["msf_topk_c5"]
    return (
        "The latency column is wall clock on one GH200 and splits three ways: "
        f"<strong>{t['select_s']:.1f} ms</strong> ranking the video, "
        f"<strong>{t['fetch_s']:.0f} ms</strong> fetching the twenty frames those "
        f"five calls look at, and <strong>{t['vlm_s']:.0f} ms</strong> in the model. "
        f"Fetching is {100*t['fetch_s']/total('msf_topk_c5'):.0f}% of it, and it is "
        "an implementation artifact rather than a cost of the method &mdash; the "
        "same frames come back in about 1 ms each from a blob store instead of "
        f"{t['fetch_s']/20:.0f} ms each from ffmpeg seeks. The retrieval-only row "
        f"is {fmt_ms(total('msf_topk'))} because it touches no pixels at all: it "
        "is a matrix multiply against an index that already exists."
    )


def build(src_rel: str, out_rel: str) -> None:
    html = open(os.path.join(ROOT, src_rel)).read()
    missing: list[str] = []

    def sub_fig(m):
        name = m.group(1)
        p = os.path.join(FIGS, f"{name}.png")
        if not os.path.exists(p):
            missing.append(name)
            return ""
        with open(p, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode()

    html = re.sub(r"\{\{FIG:([A-Za-z0-9_.\-]+)\}\}", sub_fig, html)
    if missing:
        sys.exit(f"{src_rel}: missing figures " + ", ".join(sorted(set(missing))))

    html = re.sub(r"\{\{T:([A-Za-z0-9_]+)\}\}",
                  lambda m: fmt_ms(total(m.group(1))), html)

    # {{V:run:field}} pulls a metric straight out of a run file, so a number in
    # the prose can never drift from the artifact it came from
    def sub_val(m):
        row = json.load(open(os.path.join(ROOT, TIMED[m.group(1)])))["rows"][0]
        v = row["mAP5_matched"] if m.group(2) == "mAP5" else row[m.group(2)]
        return f"{v:.2f}"

    html = re.sub(r"\{\{V:([A-Za-z0-9_]+):([A-Za-z0-9_]+)\}\}", sub_val, html)
    html = html.replace("{{LATENCY_NOTE}}", latency_note())

    left = re.findall(r"\{\{[^}]*\}\}", html)
    if left:
        sys.exit(f"{src_rel}: unresolved placeholders {sorted(set(left))}")

    out = os.path.join(ROOT, out_rel)
    open(out, "w").write(html)
    body = re.sub(r"<(style|script).*?</\1>", "", html, flags=re.S)
    words = len(re.sub(r"<[^>]+>", " ", re.sub(r'src="data:[^"]*"', "", body)).split())
    print(f"wrote {out_rel}  ({os.path.getsize(out)/1e6:.2f} MB, ~{words:,} words)")


for src, out in PAGES:
    build(src, out)
