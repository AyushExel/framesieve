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
    from .index import build_index

    if not os.path.exists(args.video):
        _err(f"no such file: {args.video}")
        return 2

    use_store = bool(args.store)
    if use_store:
        try:
            import lance  # noqa: F401
        except ImportError:
            _err("warning: --store needs `pip install pylance`; "
                 "building a plain index instead")
            use_store = False

    out = args.out or (
        f"{os.path.splitext(args.video)[0]}.framesieve-{args.encoder}-"
        f"{args.fps:g}fps.{'lance' if use_store else 'npz'}")
    if os.path.exists(out) and not args.force:
        _err(f"index already exists: {out}\n  (use --force to rebuild)")
        return 0

    from .encoders import SiglipEncoder
    enc = SiglipEncoder(args.encoder)
    _err(f"indexing {args.video}")
    _err(f"  encoder {enc.spec.repo} @ {enc.spec.revision}, {args.fps} fps")
    t0 = time.perf_counter()

    if use_store:
        # keeps the decoded frames as JPEG blobs beside their embeddings, so the
        # refine stage reads bytes instead of seeking the video: 0.9 ms a frame
        # against 12 ms across 32 workers, for about 0.3x the video in disk
        from .store import build_store
        build_store(args.video, enc, out, target_fps=args.fps, size=args.size,
                    batch=args.batch, segment_tau=args.segment_tau,
                    jpeg_quality=args.jpeg_quality, gpu_decode=args.gpu_decode,
                    seed=args.seed)
        _err(f"  wrote {out}")
        if args.json:
            print(json.dumps({"index": out, "store": True}))
        return 0

    idx = build_index(args.video, enc, target_fps=args.fps, size=args.size,
                      batch=args.batch, segment_tau=args.segment_tau,
                      pixel_gate_tau=args.pixel_gate_tau,
                      gpu_decode=args.gpu_decode, seed=args.seed,
                      verbose=not args.json)
    idx.save(out)
    dt = time.perf_counter() - t0
    mb = os.path.getsize(out) / 1e6
    hours = idx.stats.duration_s / 3600
    _err(f"  wrote {out}  ({mb:.1f} MB, {mb/max(hours,1e-9):.1f} MB per hour)")
    _err(f"  {dt:.1f} s for {hours:.2f} h of video "
         f"= {idx.stats.duration_s/max(dt,1e-9):.0f}x realtime")
    if args.json:
        print(json.dumps({"index": out, "megabytes": round(mb, 2),
                          "frames": int(idx.stats.n_frames),
                          "duration_s": round(idx.stats.duration_s, 2),
                          "wall_s": round(dt, 2)}))
    return 0


def cmd_search(args) -> int:
    from . import api

    try:
        video = api.open(args.video, encoder=args.encoder, fps=args.fps,
                         vlm=args.vlm) if args.build_missing else \
            api.load(args.video, encoder=args.encoder, fps=args.fps, vlm=args.vlm)
    except FileNotFoundError as e:
        _err(str(e))
        return 2

    hits = video.search(args.query, k=args.budget, confirm=not args.no_refine,
                        question=args.question, strategy=args.strategy,
                        tokens_per_frame=args.tokens_per_frame, seed=args.seed)

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
            best = hits[0].vlm_score if len(hits) else float("nan")
            print(f"\nno frame scored above the threshold of {args.threshold}. "
                  f"The best of the {len(hits)} frames the model was shown "
                  f"scored {best:.2f}.")
            print("  The scale is a log-odds margin: 0 is a coin flip, +2 is "
                  "about 7:1 for yes.")
            print(f"  Too strict? retry with --threshold {best-0.5:.1f}. "
                  f"Never a candidate? retry with --budget {max(8, hits.budget*4)}.")
            print("\n  what it did look at:")
    else:
        print(f"\ntop {min(args.top, len(hits))} candidates "
              f"(retrieval only; --confirm to ask the model):")

    # only show the similarity column separately when there is a VLM verdict to
    # compare it against; otherwise it is the same number printed twice
    if hits.confirmed:
        print(f"  {'time':>12}  {'hh:mm:ss':>10}  {'vlm score':>12}  "
              f"{'similarity':>11}")
        for h in shown[: args.top]:
            print(f"  {h.time:>12.1f}  {h.timecode:>10}  "
                  f"{h.vlm_score:>12.2f}  {h.score:>11.3f}")
    else:
        print(f"  {'time':>12}  {'hh:mm:ss':>10}  {'similarity':>12}")
        for h in shown[: args.top]:
            print(f"  {h.time:>12.1f}  {h.timecode:>10}  {h.score:>12.3f}")


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
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--json", action="store_true",
                        help="machine-readable output on stdout")

    p = sub.add_parser("index", parents=[common],
                       help="build the index (once per video)")
    p.add_argument("video")
    p.add_argument("--out", default=None, help="where to write the index")
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
                        "--confirm's frame fetch ~14x faster (needs pylance)")
    p.add_argument("--jpeg-quality", type=int, default=90,
                   help="quality for --store (default: %(default)s)")
    p.add_argument("--force", action="store_true", help="rebuild if it exists")
    p.set_defaults(fn=cmd_index)

    s = sub.add_parser("search", parents=[common],
                       help="search an indexed video")
    s.add_argument("video")
    s.add_argument("query", help="what to look for, phrased as a caption")
    s.add_argument("--budget", "-k", type=int, default=32,
                   help="candidate frames to consider, and model calls to spend "
                        "with --confirm (default: %(default)s)")
    s.add_argument("--confirm", dest="no_refine", action="store_false",
                   help="ask a vision-language model to check each candidate "
                        "(default; ~30 ms per frame)")
    s.add_argument("--no-refine", dest="no_refine", action="store_true",
                   help="retrieval only: no VLM, no frame fetch, ~1 ms")
    s.set_defaults(no_refine=False)
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
