"""The `framesieve` command.

    framesieve index  VIDEO
    framesieve search VIDEO "a dark tunnel" [--confirm]
    framesieve info   VIDEO

Built on the same public API as the Python side, so the two cannot drift.
`--json` on any command prints machine-readable output on stdout and keeps the
human commentary on stderr, which makes the tool usable in a pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Dependencies are checked before anything heavy is imported: a bare
# ModuleNotFoundError traceback from the wrong interpreter is the single most
# likely first experience for someone who has just cloned this.
_NEEDED = ("numpy", "torch", "transformers")


def _check_deps() -> None:
    import importlib.util
    missing = [m for m in _NEEDED if importlib.util.find_spec(m) is None]
    if missing:
        sys.exit(
            f"framesieve needs {', '.join(missing)}, and this interpreter "
            f"({sys.executable}) does not have them.\n"
            "  pip install framesieve            # or, from a clone:\n"
            "  pip install -e .")


def _timecode(t: float) -> str:
    h, rem = divmod(int(max(0.0, t)), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def _err(*a) -> None:
    print(*a, file=sys.stderr)


# --------------------------------------------------------------------------


def cmd_index(args) -> int:
    from . import api

    if not os.path.exists(args.video):
        _err(f"no such file: {args.video}")
        return 2

    import importlib.util

    use_store = bool(args.store)
    if use_store and any(importlib.util.find_spec(m) is None
                         for m in ("lance", "PIL")):
        _err("warning: --store needs `pip install \"framesieve[store]\"`; "
             "building a plain index instead")
        use_store = False

    out = args.out or api.index_path_for(args.video, args.encoder, args.fps)
    if os.path.exists(out) and not args.force:
        _err(f"index already exists: {out}\n  (use --force to rebuild)")
        return 0

    _err(f"indexing {args.video}")
    t0 = time.perf_counter()
    v = api.index(args.video, encoder=args.encoder, fps=args.fps,
                  device=args.device, store=use_store, audio=args.audio,
                  ocr=args.ocr, ocr_every=args.ocr_every,
                  language=args.language, size=args.size,
                  batch=args.batch, segment_tau=args.segment_tau,
                  pixel_gate_tau=args.pixel_gate_tau,
                  gpu_decode=args.gpu_decode, seed=args.seed,
                  jpeg_quality=args.jpeg_quality, out=args.out,
                  verbose=not args.json)
    dt = time.perf_counter() - t0
    hours = v.duration / 3600
    size_b = (sum(os.path.getsize(os.path.join(r, f))
                  for r, _, fs_ in os.walk(v.path) for f in fs_)
              if os.path.isdir(v.path) else os.path.getsize(v.path))
    mb = size_b / 1e6
    _err(f"  wrote {v.path}  ({mb:.1f} MB, {mb/max(hours,1e-9):.1f} MB per hour)")
    # two numbers, because they answer different questions: the throughput is
    # what a longer video will run at, and the wall clock includes loading the
    # model, which is a one-off that dominates on a short clip and disappears on
    # a long one. Reporting only the wall clock makes a 90-second test look slow.
    _err(f"  {v.stats.realtime_factor:.0f}x realtime encoding, "
         f"{dt:.1f} s total for {hours:.2f} h including model load")
    if v.has_speech:
        _err(f"  transcript: {len(v.speech)} segments")
    if v.has_text:
        _err(f"  on-screen text: {len(v.text)} frames carried it")
    if args.json:
        print(json.dumps({"index": v.path, "store": use_store,
                          "megabytes": round(mb, 2), "frames": len(v),
                          "duration_s": round(v.duration, 2),
                          "wall_s": round(dt, 2)}))
    return 0


def cmd_search(args) -> int:
    import importlib.util

    from . import api

    if args.confirm and importlib.util.find_spec("PIL") is None:
        # checked before the model download: --confirm pulls a 7B VLM, and
        # discovering the missing dependency after 16 GB is the wrong order
        _err("--confirm fetches frames and shows them to a vision-language "
             "model, which needs pillow:\n"
             "  pip install \"framesieve[vlm]\"")
        return 2

    try:
        opts = dict(encoder=args.encoder, fps=args.fps, vlm=args.vlm,
                    device=args.device)
        video = (api.open(args.video, **opts) if args.build_missing
                 else api.load(args.video, **opts))
    except FileNotFoundError as e:
        _err(str(e))
        return 2

    try:
        hits = video.search(args.query, k=args.budget,
                            confirm=args.confirm, source=args.source,
                            question=args.question, strategy=args.strategy,
                            tokens_per_frame=args.tokens_per_frame,
                            seed=args.seed)
    except ValueError as e:
        _err(f"error: {e}")
        return 2

    if args.json:
        print(json.dumps({
            "query": args.query, "video": str(video.video),
            "confirmed": hits.confirmed, "latency_ms": round(hits.latency_ms, 1),
            "hits": hits.to_dicts()[: args.top]}, indent=2))
    else:
        _print_hits(video, hits, args)

    if args.save_frames and len(hits):
        _save_frames(video, hits[: args.top], args.save_frames)
    # a search that confirmed nothing is a real answer, not an error
    return 0


def _print_hits(video, hits, args) -> None:
    t = hits.timings
    print(f"\nquery    : {args.query!r}")
    print(f"video    : {video.video}")
    print(f"index    : {len(video):,} frames, {video.duration/3600:.2f} h, "
          f"{video.stats.encoder}")
    print(f"strategy : {hits.strategy}, {hits.budget} candidates "
          f"({100*hits.budget/max(1,len(video)):.2f}% of frames)")
    print(f"timing   : select {1000*t.get('select_s',0):.1f} ms, "
          f"fetch {t.get('fetch_s',0):.2f} s, vlm {t.get('vlm_s',0):.2f} s")

    shown = hits
    if hits.confirmed:
        kept = hits.above(args.threshold)
        if len(kept):
            shown = kept
            print(f"\n{len(kept)} hit(s) above threshold {args.threshold}:")
        else:
            # non-visual hits carry no VLM score, and one of them can be first
            scored = [h.vlm_score for h in hits if h.vlm_score is not None]
            best = max(scored) if scored else float("nan")
            print(f"\nno frame scored above the threshold of {args.threshold}. "
                  f"The best of the {len(scored)} frames the model was shown "
                  f"scored {best:.2f}.")
            print("  The scale is a log-odds margin: 0 is a coin flip, +2 is "
                  "about 7:1 for yes.")
            if scored:
                print(f"  Too strict? retry with --threshold {best-0.5:.1f}. "
                      f"Never a candidate? retry with --budget "
                      f"{max(8, hits.budget*4)}.")
            print("\n  what it did look at:")
    else:
        print(f"\ntop {min(args.top, len(hits))} candidates "
              f"(retrieval only; --confirm to ask the model):")

    # only show the similarity column separately when there is a VLM verdict to
    # compare it against; otherwise it is the same number printed twice
    mixed = any(h.source != "visual" for h in shown)
    cols = f"  {'time':>12}  {'hh:mm:ss':>10}"
    if hits.confirmed:
        cols += f"  {'vlm score':>12}"
    cols += f"  {'similarity':>11}"
    if mixed:
        cols += f"  {'source':<8}  what was said"
    print(cols)
    for h in shown[: args.top]:
        row = f"  {h.time:>12.1f}  {h.timecode:>10}"
        if hits.confirmed:
            row += f"  {'' if h.vlm_score is None else f'{h.vlm_score:.2f}':>12}"
        row += f"  {h.score:>11.3f}"
        if mixed:
            row += f"  {h.source:<8}  {(h.text or '')[:52]}"
        print(row)


def _save_frames(video, hits, out_dir: str) -> None:
    try:
        from PIL import Image
    except ImportError:
        _err("--save-frames needs pillow: pip install pillow")
        return
    os.makedirs(out_dir, exist_ok=True)
    frames = video.frames(hits)
    for h, fr in zip(hits, frames):
        Image.fromarray(fr).save(os.path.join(out_dir, f"t{h.time:09.1f}.jpg"),
                                 quality=92)
    _err(f"\nwrote {len(frames)} frames to {out_dir}/")


def cmd_info(args) -> int:
    from dataclasses import asdict

    from . import api
    try:
        video = api.load(args.video, encoder=args.encoder, fps=args.fps)
    except FileNotFoundError as e:
        _err(str(e))
        return 2
    d = asdict(video.stats)
    if args.json:
        print(json.dumps(d, indent=2))
        return 0
    print(f"\n{video}")
    width = max(len(k) for k in d)
    for k, v in d.items():
        print(f"  {k:<{width}}  {v}")
    return 0


# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    from . import __version__
    from .encoders import SIGLIP_MODELS
    from .search import STRATEGIES

    ap = argparse.ArgumentParser(
        prog="framesieve",
        description="Search long video without running a VLM on every frame.",
        epilog="docs: https://github.com/AyushExel/framesieve  |  "
               "index once, then every query is a matrix multiply",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version",
                    version=f"framesieve {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="COMMAND")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--encoder", default="siglip2-base-224",
                        choices=list(SIGLIP_MODELS),
                        help="the cheap per-frame encoder (default: %(default)s)")
    common.add_argument("--fps", type=float, default=1.0,
                        help="frames sampled per second of video "
                             "(default: %(default)s)")
    common.add_argument("--device", default=None,
                        help="cuda / mps / cpu (default: whichever is there)")
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--json", action="store_true",
                        help="machine-readable output on stdout")

    p = sub.add_parser("index", parents=[common],
                       help="build the index (once per video)")
    p.add_argument("video")
    p.add_argument("--out", default=None,
                   help="where to write the index (default: next to the video; "
                        "an index at a custom path is searched by passing that "
                        "path, since bare `framesieve search VIDEO` only looks "
                        "next to the video)")
    p.add_argument("--size", type=int, default=256,
                   help="decode at this resolution before encoding")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--segment-tau", type=float, default=0.90,
                   help="cosine similarity below which a new segment starts; "
                        "0 disables redundancy collapse (default: %(default)s)")
    p.add_argument("--pixel-gate-tau", type=float, default=0.0,
                   help="skip encoding frames within this mean grey-level "
                        "difference of the last kept one; 0 disables")
    p.add_argument("--gpu-decode", action="store_true",
                   help="decode with NVDEC: slower in wall clock on most hosts "
                        "but uses ~0.5 CPU cores instead of ~16")
    p.add_argument("--store", action="store_true",
                   help="also keep every sampled frame as a JPEG blob in a "
                        "Lance dataset: ~0.3x the video in disk, and makes "
                        "--confirm's frame fetch ~15x faster, and lets the "
                        "index work without the video. Costs ~250x the disk "
                        "of a plain index (needs framesieve[store])")
    p.add_argument("--jpeg-quality", type=int, default=90,
                   help="quality for --store (default: %(default)s)")
    p.add_argument("--audio", action="store_true",
                   help="also transcribe the audio with Whisper and index the "
                        "timed segments, so `search --source speech` can reach "
                        "what was said. ~11x realtime (needs framesieve[audio])")
    p.add_argument("--ocr", action="store_true",
                   help="also read the text on screen and index it, so "
                        "`search --source text` can reach a caption or a slide "
                        "title. ~1.5 min per hour of video with the default "
                        "--ocr-every segment (needs framesieve[ocr])")
    p.add_argument("--ocr-every", default="segment", choices=["segment", "frame"],
                   help="read one frame per shot (default) or every frame; "
                        "'frame' is ~4x slower and only helps when the text "
                        "changes under a still picture")
    p.add_argument("--language", default=None,
                   help="force a transcription language, e.g. en; "
                        "auto-detected otherwise")
    p.add_argument("--force", action="store_true", help="rebuild if it exists")
    p.set_defaults(fn=cmd_index)

    s = sub.add_parser("search", parents=[common],
                       help="search an indexed video")
    s.add_argument("video")
    s.add_argument("query", help="what to look for, phrased as a caption")
    s.add_argument("--budget", "-k", type=int, default=32,
                   help="candidate frames to consider, and model calls to spend "
                        "with --confirm (default: %(default)s)")
    # retrieval-only by default, exactly like the Python API's confirm=False:
    # a bare `framesieve search` should never download a 7B model unasked
    s.add_argument("--confirm", dest="confirm", action="store_true",
                   help="ask a vision-language model to check each candidate: "
                        "~30 ms per frame on a GPU, ~16 GB of weights on first "
                        "use (needs framesieve[vlm])")
    s.add_argument("--no-refine", dest="confirm", action="store_false",
                   help="retrieval only: no VLM, no frame fetch, milliseconds "
                        "(this is the default)")
    s.set_defaults(confirm=False)
    s.add_argument("--question", default=None,
                   help="the yes/no question put to the model "
                        "(default: 'Does this frame show: QUERY?')")
    s.add_argument("--strategy", default="segment_adaptive", choices=list(STRATEGIES),
                   help="how candidates are spread over the video "
                        "(default: %(default)s)")
    s.add_argument("--threshold", type=float, default=0.0,
                   help="log-odds above which a confirmed hit is reported; "
                        "0 is a coin flip (default: %(default)s)")
    s.add_argument("--top", type=int, default=20, help="rows to print")
    s.add_argument("--vlm", default="qwen2.5-vl-7b")
    s.add_argument("--tokens-per-frame", type=int, default=64,
                   help="visual tokens per frame for the VLM; lower is cheaper")
    s.add_argument("--save-frames", default=None, metavar="DIR",
                   help="write the reported frames as JPEGs")
    s.add_argument("--source", default=None,
                   choices=["visual", "speech", "text"],
                   help="search frames, the transcript, or the on-screen text; "
                        "by default whatever the index has")
    s.add_argument("--build-missing", action="store_true",
                   help="index the video first if it has no index yet")
    s.set_defaults(fn=cmd_search)

    i = sub.add_parser("info", parents=[common],
                       help="show how an index was built")
    i.add_argument("video")
    i.set_defaults(fn=cmd_info)
    return ap


def main(argv: list | None = None) -> int:
    _check_deps()
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args) or 0)
    except KeyboardInterrupt:
        _err("\ninterrupted")
        return 130
    except FileNotFoundError as e:
        _err(f"error: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
