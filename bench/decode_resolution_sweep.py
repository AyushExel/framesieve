"""How decode cost scales with resolution, content held constant.

The native test video is 960x720. Rather than swap in a different film to get a
1080p number -- which would change content, encoder settings and bitrate all at
once -- the same 300 s segment is re-encoded at five resolutions with identical
x264 settings (preset medium, CRF 21, GOP 50). Bitrate is then free to follow
content complexity the way it does in the wild.

This is a synthetic *resolution* probe on real *content*. It answers "what does
decode cost at 1080p?" without pretending the upscaled pixels carry new detail.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from decode_bench import Result, env_report, fmt_table, probe, run_case  # noqa: E402


def main() -> None:
    clips = sorted(glob.glob("data/synth/res_*.mp4"),
                   key=lambda p: int(os.path.basename(p).split("_")[1].split("x")[0]))
    if not clips:
        sys.exit("no clips in data/synth -- run the encode step first")

    rows: list[Result] = []
    print(f"{'resolution':<12}{'Mbps':>7}{'backend':>9}{'frame/s':>11}{'xRT':>9}"
          f"{'cores':>7}{'24h idx':>10}")
    print("-" * 65)
    for path in clips:
        info = probe(path)
        res = f"{info.width}x{info.height}"
        for gpu, label in ((False, "cpu"), (True, "nvdec")):
            best = None
            for _ in range(3):
                r = run_case(f"{res} {label}", info=info, gpu=gpu, strategy="all",
                             sink="null", out_wh=(384, 384), target_fps=1.0,
                             start_s=0.0, duration_s=info.duration_s,
                             threads=os.cpu_count() or 16)
                if best is None or (r.ok and r.realtime_factor > best.realtime_factor):
                    best = r
            rows.append(best)
            idx = f"{best.hours_per_24h:.2f} h" if best.ok else "FAIL"
            print(f"{res:<12}{info.bit_rate_mbps:>7.1f}{label:>9}"
                  f"{best.frames_per_s:>11.0f}{best.realtime_factor:>9.1f}"
                  f"{best.cpu_cores_used:>7.1f}{idx:>10}")

    # scaling exponent: how much does doubling pixel count cost?
    print()
    cpu = [(probe(p).width * probe(p).height, r.frames_per_s)
           for p, r in zip(clips, rows[0::2]) if r.ok]
    if len(cpu) >= 2:
        import math
        px0, f0 = cpu[0]
        px1, f1 = cpu[-1]
        expo = math.log(f0 / f1) / math.log(px1 / px0)
        print(f"CPU decode scaling: {f0:.0f} frame/s at {px0/1e6:.2f} MP -> "
              f"{f1:.0f} frame/s at {px1/1e6:.2f} MP")
        print(f"  frames/s ~ pixels^-{expo:.2f}  (1.00 would be perfectly "
              f"pixel-bound)")

    with open("runs/decode_resolution_sweep.json", "w") as f:
        json.dump({"env": env_report(),
                   "results": [r.__dict__ for r in rows]}, f, indent=2)
    print("\nwrote runs/decode_resolution_sweep.json")


if __name__ == "__main__":
    main()
