"""Indexing cost of the MomentSeeker retrieval baselines, measured not quoted.

The published table reports accuracy and parameter counts. It does not report
what any of these methods cost to run, which makes it an accuracy table rather
than a cost/accuracy table. This measures the axis that actually differs.

For *every* retrieval method here, answering a query is a dot product against a
precomputed index -- about a millisecond. What differs by orders of magnitude is
building that index: how many frames each method pushes through what size of
encoder for every ten seconds of video.

Configurations follow the paper's own "#Frames" column (frames per clip for
retrieval-based methods):

    LanguageBind   428M   8 frames/clip   video tower with temporal attention
    InternVideo2     1B   8 frames/clip
    framesieve      93M  10 frames/clip   1 fps over a 10 s chunk, image tower

Reported in the only unit that makes them comparable: GPU-seconds to index one
hour of video, and the equivalent realtime factor.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

CHUNK_S = 10.0


def timed(fn, iters: int, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def bench_languagebind(batch_chunks: int, iters: int,
                       attn_impl: str = "sdpa") -> dict:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from languagebind.video.configuration_video import LanguageBindVideoConfig
    from languagebind.video.modeling_video import LanguageBindVideo

    import _lb_shim  # noqa: F401

    # the checkpoint's model_type is not registered with AutoConfig, so the
    # config has to be built explicitly or from_pretrained resolves it to None
    repo = "LanguageBind/LanguageBind_Video_FT"
    cfg = LanguageBindVideoConfig.from_pretrained(repo)
    # LanguageBind builds its temporal-attention blocks from vision_config, which
    # in newer transformers must carry an explicit attention implementation or
    # the dispatch table lookup fails with KeyError: None
    # the class does not declare sdpa support, so eager it is -- and eager is
    # what the published numbers were produced with anyway
    for c in (cfg, cfg.vision_config, cfg.text_config):
        c._attn_implementation = "eager"
    model = LanguageBindVideo.from_pretrained(
        repo, config=cfg, dtype=torch.float16).cuda().eval()

    # from_pretrained refuses sdpa for this class, but the attention modules
    # dispatch on their own config at call time, so flip them afterwards and
    # report the faster path -- measuring the baseline on eager while measuring
    # ours on sdpa would understate it.
    def set_attn(impl):
        for m in model.modules():
            c = getattr(m, "config", None)
            if c is not None and hasattr(c, "_attn_implementation"):
                c._attn_implementation = impl

    set_attn(attn_impl)
    nf = model.config.vision_config.num_frames
    n_params = sum(p.numel() for p in model.parameters())
    n_vision = sum(p.numel() for p in model.vision_model.parameters())

    # (batch, channels, frames, height, width) is what the video tower expects
    x = torch.randn(batch_chunks, 3, nf, 224, 224, dtype=torch.float16, device="cuda")

    @torch.inference_mode()
    def run():
        model.get_image_features(pixel_values=x)

    per_call = timed(run, iters)
    chunks_per_s = batch_chunks / per_call
    return dict(name="LanguageBind", attn=attn_impl, params_m=n_params / 1e6,
                vision_params_m=n_vision / 1e6, frames_per_chunk=nf,
                batch_chunks=batch_chunks, s_per_batch=per_call,
                chunks_per_s=chunks_per_s,
                video_s_per_s=chunks_per_s * CHUNK_S,
                gpu_s_per_hour=3600.0 / (chunks_per_s * CHUNK_S))


def bench_framesieve(batch_chunks: int, iters: int, frames_per_chunk: int = 10) -> dict:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from framesieve.encoders import SiglipEncoder

    enc = SiglipEncoder("siglip2-base-224")
    d = enc.describe()
    n = batch_chunks * frames_per_chunk
    x = torch.randint(0, 255, (n, 224, 224, 3), dtype=torch.uint8).pin_memory()

    def run():
        enc.encode_frames(x)

    per_call = timed(run, iters)
    chunks_per_s = batch_chunks / per_call
    return dict(name="framesieve (SigLIP2-base)", params_m=d["params_total_m"],
                vision_params_m=d["params_vision_m"],
                frames_per_chunk=frames_per_chunk, batch_chunks=batch_chunks,
                s_per_batch=per_call, chunks_per_s=chunks_per_s,
                video_s_per_s=chunks_per_s * CHUNK_S,
                gpu_s_per_hour=3600.0 / (chunks_per_s * CHUNK_S))


def bench_internvideo2(batch_chunks: int, iters: int) -> dict:
    """InternVideo2's released checkpoint needs its training repo to instantiate.

    Rather than skip the frontier model, its vision tower is reconstructed from
    the published architecture -- a 1B ViT (dim 1408, depth 40, heads 16,
    patch 14) over 8 frames at 224 -- using timm's building blocks. That measures
    the compute the architecture implies, which is the quantity being compared,
    and it is labelled as reconstructed rather than run from the checkpoint.
    """
    try:
        from timm.models.vision_transformer import Block
    except ImportError:
        return dict(name="InternVideo2", error="timm not installed")

    import torch.nn as nn

    dim, depth, heads, patch, nf = 1408, 40, 16, 14, 8
    tokens = (224 // patch) ** 2 * nf + 1

    class Tower(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Conv3d(3, dim, kernel_size=(1, patch, patch),
                                  stride=(1, patch, patch))
            self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
            self.norm = nn.LayerNorm(dim)

        def forward(self, x):
            x = self.proj(x).flatten(2).transpose(1, 2)
            for b in self.blocks:
                x = b(x)
            return self.norm(x).mean(1)

    model = Tower().half().cuda().eval()
    n_params = sum(p.numel() for p in model.parameters())
    x = torch.randn(batch_chunks, 3, nf, 224, 224, dtype=torch.float16, device="cuda")

    @torch.inference_mode()
    def run():
        model(x)

    per_call = timed(run, iters)
    chunks_per_s = batch_chunks / per_call
    return dict(name="InternVideo2 (reconstructed)", params_m=n_params / 1e6,
                vision_params_m=n_params / 1e6, frames_per_chunk=nf,
                batch_chunks=batch_chunks, s_per_batch=per_call,
                chunks_per_s=chunks_per_s, video_s_per_s=chunks_per_s * CHUNK_S,
                gpu_s_per_hour=3600.0 / (chunks_per_s * CHUNK_S),
                note="architecture reconstructed from the paper, not the released checkpoint",
                tokens_per_chunk=tokens)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-chunks", type=int, default=16)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--which", nargs="*",
                    default=["framesieve", "languagebind", "internvideo2"])
    ap.add_argument("--out", default="runs/baseline_throughput.json")
    args = ap.parse_args()

    rows = []
    fns = {"framesieve": bench_framesieve, "languagebind": bench_languagebind,
           "internvideo2": bench_internvideo2}
    for k in args.which:
        print(f"benchmarking {k} ...", flush=True)
        try:
            r = fns[k](args.batch_chunks, args.iters)
        except Exception as e:  # noqa: BLE001
            r = dict(name=k, error=f"{type(e).__name__}: {str(e)[:200]}")
        rows.append(r)
        torch.cuda.empty_cache()
        if "error" in r:
            print(f"  FAILED: {r['error']}")
        else:
            print(f"  {r['vision_params_m']:.0f}M vision params, "
                  f"{r['frames_per_chunk']} frames/chunk -> "
                  f"{r['video_s_per_s']:.0f}x realtime, "
                  f"{r['gpu_s_per_hour']:.1f} GPU-s per hour of video")

    print(f"\n{'method':<32}{'vision M':>10}{'fr/chunk':>10}"
          f"{'xRT':>9}{'GPU-s / h video':>18}")
    print("-" * 79)
    for r in rows:
        if "error" in r:
            print(f"{r['name']:<32}{'--':>10}{'--':>10}{'--':>9}{r['error'][:30]:>18}")
            continue
        print(f"{r['name']:<32}{r['vision_params_m']:>10.0f}"
              f"{r['frames_per_chunk']:>10}{r['video_s_per_s']:>9.0f}"
              f"{r['gpu_s_per_hour']:>18.1f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(config=vars(args), chunk_s=CHUNK_S,
                       gpu=torch.cuda.get_device_name(0), rows=rows), f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
