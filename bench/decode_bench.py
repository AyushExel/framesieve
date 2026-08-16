"""Decode throughput reality check.

The premise of a cheap-dense-index cascade is that we can push every frame of a
long video through a small vision encoder. That premise dies if we cannot even
*decode* frames fast enough. This measures the decode ceiling on real footage.

Three axes are varied, because all three matter and get conflated:

  backend   : who does the H.264 work -- libavcodec on CPU vs NVDEC on the GPU
  strategy  : how you get down to ~1 fps -- decode-everything-and-drop,
              keyframe-only, or seek-to-timestamp
  sink      : where the pixels end up -- nowhere (pure decode ceiling), host RAM
              through a pipe, or a CUDA tensor

The headline number is not frames/sec, it is the *realtime factor*: seconds of
video chewed through per second of wall clock. That says whether indexing a 24 h
video takes 20 minutes or 20 hours.

Everything here is measured. Nothing is extrapolated except the explicitly
labelled "hours to index 24 h" column, which is just 24 / realtime_factor.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"


# --------------------------------------------------------------------------
# probing
# --------------------------------------------------------------------------


@dataclass
class VideoInfo:
    path: str
    duration_s: float
    fps: float
    width: int
    height: int
    codec: str
    pix_fmt: str
    bit_rate_mbps: float
    n_frames_est: int
    size_gb: float

    def pretty(self) -> str:
        h, m = int(self.duration_s // 3600), int((self.duration_s % 3600) // 60)
        return (f"{os.path.basename(self.path)}  {h}h{m:02d}m  {self.width}x{self.height}  "
                f"{self.codec} {self.pix_fmt}  {self.fps:.3f} fps  "
                f"{self.bit_rate_mbps:.1f} Mbps  {self.size_gb:.2f} GB  "
                f"~{self.n_frames_est:,} frames")


def probe(path: str) -> VideoInfo:
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name,width,height,pix_fmt,avg_frame_rate,r_frame_rate,nb_frames",
         "-show_entries", "format=duration,bit_rate", "-of", "json", path],
        capture_output=True, text=True, check=True)
    d = json.loads(out.stdout)
    st, fmt = d["streams"][0], d["format"]

    def _rate(s: str) -> float:
        if not s or s == "0/0":
            return 0.0
        a, _, b = s.partition("/")
        return float(a) / float(b) if b and float(b) else float(a)

    fps = _rate(st.get("avg_frame_rate", "")) or _rate(st.get("r_frame_rate", ""))
    duration = float(fmt.get("duration", 0.0))
    nb = st.get("nb_frames")
    return VideoInfo(
        path=path, duration_s=duration, fps=fps,
        width=int(st["width"]), height=int(st["height"]),
        codec=st["codec_name"], pix_fmt=st.get("pix_fmt", "?"),
        bit_rate_mbps=float(fmt.get("bit_rate", 0)) / 1e6,
        n_frames_est=int(nb) if nb and nb.isdigit() else int(round(duration * fps)),
        size_gb=os.path.getsize(path) / 1e9)


def keyframe_stats(path: str, window_s: int = 300) -> dict:
    """Keyframe spacing bounds what a keyframe-only strategy can deliver.

    If keyframes sit 2 s apart you cannot get 1 fps from them at any speed -- the
    frames simply are not there. Measured from packet flags, which is exact and
    does not require decoding.
    """
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-read_intervals", f"%+{int(window_s)}",
         "-select_streams", "v:0", "-show_entries", "packet=pts_time,flags",
         "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True)
    ts = []
    for line in out.stdout.splitlines():
        parts = line.strip().split(",")
        if len(parts) >= 2 and "K" in parts[1]:
            try:
                ts.append(float(parts[0]))
            except ValueError:
                pass
    if len(ts) < 2:
        return {"n_keyframes": len(ts), "mean_gop_s": float("inf"), "keyframes_per_s": 0.0}
    gaps = np.diff(sorted(ts))
    return {"n_keyframes": len(ts), "mean_gop_s": float(gaps.mean()),
            "max_gop_s": float(gaps.max()), "keyframes_per_s": float(1.0 / gaps.mean())}


# --------------------------------------------------------------------------
# ffmpeg command construction
# --------------------------------------------------------------------------


def build_cmd(path: str, *, gpu: bool, strategy: str, sink: str, out_wh, target_fps: float,
              start_s: float, duration_s: float, threads: int,
              cuvid_resize: bool = False) -> list[str]:
    W, H = out_wh
    cmd = [FFMPEG, "-hide_banner", "-nostdin", "-v", "error"]
    if sink == "null":
        cmd += ["-stats"]

    if strategy == "keyframe":
        # tell libavcodec to discard everything that is not an I-frame; on the CPU
        # path this skips inter-frame reconstruction entirely
        cmd += ["-skip_frame", "nokey"]

    if start_s > 0:
        cmd += ["-ss", f"{start_s}"]

    if gpu:
        cmd += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-c:v", "h264_cuvid"]
        if cuvid_resize:
            cmd += ["-resize", f"{W}x{H}"]   # scale inside the decoder: cheapest path
    else:
        cmd += ["-threads", str(threads)]

    cmd += ["-i", path]
    if duration_s > 0:
        cmd += ["-t", f"{duration_s}"]
    cmd += ["-an", "-sn"]                     # audio/subs are pure overhead here

    filters = []
    if strategy == "fps" and target_fps > 0:
        filters.append(f"fps={target_fps}")
    if sink == "pipe":
        if gpu:
            if not cuvid_resize:
                filters.append(f"scale_cuda={W}:{H}")
            filters += ["hwdownload", "format=nv12"]
        else:
            filters.append(f"scale={W}:{H}")

    if strategy in ("keyframe", "all") or (strategy == "fps" and sink == "null"):
        # never let the muxer duplicate frames back up to the source rate.
        # ffmpeg 4.4 spells this -vsync 0; -fps_mode does not exist yet.
        cmd += ["-vsync", "0"]

    if filters:
        cmd += ["-vf", ",".join(filters)]

    if sink == "null":
        cmd += ["-f", "null", "-"]
    else:
        cmd += ["-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    return cmd


_FRAME_RE = re.compile(rb"frame=\s*(\d+)")


def run_null_sink(cmd: list[str], timeout: float) -> tuple[int, float, str]:
    """Pure decode ceiling: ffmpeg decodes and throws the pixels away."""
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, timeout=timeout)
    dt = time.perf_counter() - t0
    m = _FRAME_RE.findall(p.stderr)
    n = int(m[-1]) if m else 0
    err = p.stderr.decode("utf8", "ignore")
    return n, dt, err


def run_pipe_sink(cmd: list[str], frame_bytes: int, timeout: float) -> tuple[int, float, str]:
    """Delivered pixels: frames actually land in host RAM as numpy-readable bytes."""
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         bufsize=frame_bytes * 8)
    n = 0
    try:
        while True:
            if time.perf_counter() - t0 > timeout:
                break
            buf = p.stdout.read(frame_bytes)
            if not buf or len(buf) < frame_bytes:
                break
            n += 1
    finally:
        try:
            p.stdout.close()
        except Exception:
            pass
        p.terminate()
        try:
            err = p.stderr.read().decode("utf8", "ignore")
            p.stderr.close()
        except Exception:
            err = ""
        try:
            p.wait(timeout=20)
        except Exception:
            p.kill()
    return n, time.perf_counter() - t0, err


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------


@dataclass
class Result:
    name: str
    backend: str
    strategy: str
    sink: str
    workers: int = 1
    ok: bool = False
    frames: int = 0
    wall_s: float = 0.0
    frames_per_s: float = 0.0
    video_s: float = 0.0
    realtime_factor: float = 0.0
    hours_per_24h: float = 0.0
    cpu_cores_used: float = 0.0
    note: str = ""
    cmd: str = ""


def finish(r: Result, n: int, dt: float, video_s: float, err: str) -> Result:
    r.frames, r.wall_s = n, dt
    r.ok = n > 0
    r.frames_per_s = n / dt if dt else 0.0
    r.video_s = video_s
    r.realtime_factor = video_s / dt if dt else 0.0
    r.hours_per_24h = 24.0 / r.realtime_factor if r.realtime_factor else float("inf")
    if not r.ok:
        r.note = err.strip().splitlines()[-1][:200] if err.strip() else "no frames"
    return r


def run_case(name: str, info: VideoInfo, *, gpu: bool, strategy: str, sink: str,
             out_wh, target_fps: float, start_s: float, duration_s: float,
             threads: int, cuvid_resize: bool = False, workers: int = 1,
             timeout: float = 900.0) -> Result:
    """One measurement. With workers>1 the window is split into disjoint chunks
    processed by independent ffmpeg processes -- this is how you would really use
    a 64-core CPU or a GPU with several NVDEC engines."""
    W, H = out_wh
    r = Result(name=name, backend="nvdec" if gpu else "cpu", strategy=strategy,
               sink=sink, workers=workers)
    chunk = duration_s / workers
    cmds = [build_cmd(info.path, gpu=gpu, strategy=strategy, sink=sink, out_wh=out_wh,
                      target_fps=target_fps, start_s=start_s + i * chunk,
                      duration_s=chunk, threads=max(1, threads // workers),
                      cuvid_resize=cuvid_resize)
            for i in range(workers)]
    r.cmd = " ".join(cmds[0])

    cpu0 = time.process_time()
    ru0 = os.times()
    t0 = time.perf_counter()
    if workers == 1:
        n, dt, err = (run_null_sink(cmds[0], timeout) if sink == "null"
                      else run_pipe_sink(cmds[0], W * H * 3, timeout))
    else:
        fn = ((lambda c: run_null_sink(c, timeout)) if sink == "null"
              else (lambda c: run_pipe_sink(c, W * H * 3, timeout)))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            outs = list(ex.map(fn, cmds))
        n = sum(o[0] for o in outs)
        err = next((o[2] for o in outs if o[0] == 0), "")
        dt = time.perf_counter() - t0
    ru1 = os.times()
    r.cpu_cores_used = ((ru1.children_user - ru0.children_user) +
                        (ru1.children_system - ru0.children_system)) / dt if dt else 0.0
    return finish(r, n, dt, duration_s, err)


def fmt_table(rows: list[Result]) -> str:
    hdr = (f"{'case':<38}{'sink':<7}{'wrk':>4}{'frames':>9}{'wall s':>8}"
           f"{'frame/s':>10}{'xRT':>9}{'cores':>7}{'24h idx':>10}")
    out = [hdr, "-" * len(hdr)]
    for r in rows:
        if not r.ok:
            out.append(f"{r.name:<38}{r.sink:<7}{r.workers:>4}{'FAILED':>9}"
                       f"{r.wall_s:>8.1f}{'':>10}{'':>9}{'':>7}{'':>10}  {r.note[:50]}")
            continue
        idx = f"{r.hours_per_24h:.2f} h" if r.hours_per_24h < 1e5 else "inf"
        out.append(f"{r.name:<38}{r.sink:<7}{r.workers:>4}{r.frames:>9,}{r.wall_s:>8.1f}"
                   f"{r.frames_per_s:>10.0f}{r.realtime_factor:>9.1f}"
                   f"{r.cpu_cores_used:>7.1f}{idx:>10}")
    return "\n".join(out)


def env_report() -> dict:
    def _sh(c):
        try:
            return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
        except Exception:
            return "?"
    return {"platform": platform.platform(), "machine": platform.machine(),
            "python": sys.version.split()[0], "cpu_count": os.cpu_count(),
            "ffmpeg": _sh("ffmpeg -version 2>/dev/null | head -1"),
            "gpu": _sh("nvidia-smi --query-gpu=name,memory.total,driver_version "
                       "--format=csv,noheader"),
            "nvcuvid_present": _sh("ldconfig -p | grep -c nvcuvid")}


# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--seconds", type=float, default=1200.0)
    ap.add_argument("--start", type=float, default=600.0)
    ap.add_argument("--size", type=int, nargs=2, default=(384, 384))
    ap.add_argument("--target-fps", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 16)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--scaling", type=int, nargs="*", default=[1, 2, 4, 8],
                    help="worker counts for the parallel scaling sweep")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    info = probe(args.video)
    kf = keyframe_stats(args.video)
    print("VIDEO :", info.pretty())
    print(f"KEYFR : {kf['n_keyframes']} keyframes in first 300 s -> mean GOP "
          f"{kf['mean_gop_s']:.2f} s = {kf['keyframes_per_s']:.3f} keyframes/s"
          f"  (max gap {kf.get('max_gop_s', 0):.1f} s)")
    if kf["keyframes_per_s"] < args.target_fps:
        print(f"        NOTE: keyframe-only cannot reach {args.target_fps} fps on this "
              f"video -- it tops out at {kf['keyframes_per_s']:.3f} fps.")
    print()

    common = dict(info=info, out_wh=tuple(args.size), target_fps=args.target_fps,
                  start_s=args.start, duration_s=args.seconds, threads=args.threads)

    cases: list[tuple[str, dict]] = [
        # ---- pure decode ceiling: no pixels delivered anywhere ---------------
        ("cpu  decode-all", dict(gpu=False, strategy="all", sink="null")),
        ("nvdec decode-all", dict(gpu=True, strategy="all", sink="null")),
        ("cpu  keyframes-only", dict(gpu=False, strategy="keyframe", sink="null")),
        # ---- pixels delivered to host RAM at model input size ----------------
        ("cpu  all->1fps  ->host", dict(gpu=False, strategy="fps", sink="pipe")),
        ("nvdec all->1fps ->host", dict(gpu=True, strategy="fps", sink="pipe")),
        ("nvdec all->1fps ->host (dec-resize)",
         dict(gpu=True, strategy="fps", sink="pipe", cuvid_resize=True)),
        ("cpu  keyframes  ->host", dict(gpu=False, strategy="keyframe", sink="pipe")),
        ("cpu  decode-all ->host", dict(gpu=False, strategy="all", sink="pipe")),
        ("nvdec decode-all ->host", dict(gpu=True, strategy="all", sink="pipe")),
    ]

    results: list[Result] = []
    print("--- single worker ---")
    for name, kw in cases:
        best: Result | None = None
        for _ in range(args.repeats):
            r = run_case(name, **common, **kw)
            if best is None or (r.ok and r.realtime_factor > best.realtime_factor):
                best = r
        assert best is not None
        results.append(best)
        print(f"  [{'ok  ' if best.ok else 'FAIL'}] {name:<38}"
              f"{best.realtime_factor:>8.1f} xRT  {best.frames:>8,} fr  "
              f"{best.cpu_cores_used:>5.1f} cores"
              + ("" if best.ok else f"   {best.note[:70]}"))

    # ---- how far does this scale with independent workers? -------------------
    print("\n--- parallel scaling (decode-all, null sink) ---")
    for gpu, label in ((False, "cpu"), (True, "nvdec")):
        for w in args.scaling:
            r = run_case(f"{label} decode-all x{w}", **common, gpu=gpu,
                         strategy="all", sink="null", workers=w)
            results.append(r)
            print(f"  [{'ok  ' if r.ok else 'FAIL'}] {label:<6} workers={w:<2} "
                  f"{r.realtime_factor:>8.1f} xRT   {r.frames_per_s:>8.0f} frame/s  "
                  f"{r.cpu_cores_used:>5.1f} cores")

    print()
    print(fmt_table(results))

    payload = {"video": asdict(info), "keyframes": kf, "config": vars(args),
               "env": env_report(), "results": [asdict(r) for r in results]}
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
