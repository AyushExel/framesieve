"""The expensive stage: a real VLM asked a closed question about a frame.

Two design choices here matter for everything downstream.

1. We score, we do not chat. Asking Qwen "is there a red car here?" and parsing
   the string gives a bit per frame and no way to draw a recall curve. Instead we
   read the logits of the "Yes" and "No" tokens at the first generated position
   and take their difference. That is one forward pass, no sampling, and it
   yields a continuous, monotone score you can threshold anywhere.

2. Visual tokens are the cost unit. Qwen2.5-VL uses dynamic resolution, so the
   per-frame cost is set by `max_pixels`, not by the source video. We expose it
   and measure against it, because "VLM cost per frame" is meaningless without it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

QWEN_MODELS: dict[str, dict] = {
    "qwen2.5-vl-7b": dict(repo="Qwen/Qwen2.5-VL-7B-Instruct", revision="cc594898137f"),
    "qwen2.5-vl-3b": dict(repo="Qwen/Qwen2.5-VL-3B-Instruct", revision="66285546d2b8"),
}

# 28x28 pixels per visual token before the 2x2 spatial merge in Qwen2.5-VL.
QWEN_PATCH = 28
QWEN_MERGE = 2


@dataclass
class VlmSpec:
    key: str
    repo: str
    revision: str
    max_pixels: int
    min_pixels: int


class QwenYesNoScorer:
    """Scores frames against a yes/no question with a single forward pass each."""

    PROMPT = ("Answer with exactly one word, Yes or No.\n"
              "Question: {q}")

    def __init__(self, key: str = "qwen2.5-vl-7b", device: str = "cuda",
                 dtype: torch.dtype = torch.bfloat16,
                 max_pixels: int = 256 * 28 * 28, min_pixels: int = 64 * 28 * 28,
                 attn_impl: str | None = None):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        cfg = QWEN_MODELS[key]
        self.spec = VlmSpec(key=key, repo=cfg["repo"], revision=cfg["revision"],
                            max_pixels=max_pixels, min_pixels=min_pixels)
        self.device = device
        if attn_impl is None:
            attn_impl = "flash_attention_2" if _has_flash_attn() else "sdpa"
        self.attn_impl = attn_impl

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            cfg["repo"], revision=cfg["revision"], dtype=dtype,
            attn_implementation=attn_impl,
        ).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(
            cfg["repo"], revision=cfg["revision"],
            min_pixels=min_pixels, max_pixels=max_pixels,
        )
        self.tok = self.processor.tokenizer
        self.tok.padding_side = "left"          # required for batched decoding
        self.yes_ids, self.no_ids = self._answer_token_ids()

    def _answer_token_ids(self) -> tuple[list[int], list[int]]:
        """Token ids that count as Yes / No at the first generated position.

        Several surface forms map to different ids; we pool over all of them so
        the score does not depend on which capitalisation the model prefers.
        """
        def ids_for(words: Sequence[str]) -> list[int]:
            out = set()
            for w in words:
                for form in (w, " " + w):
                    enc = self.tok.encode(form, add_special_tokens=False)
                    if enc:
                        out.add(enc[0])
            return sorted(out)

        return ids_for(["Yes", "yes", "YES"]), ids_for(["No", "no", "NO"])

    def _build_inputs(self, frames: Sequence[np.ndarray], question: str):
        from PIL import Image
        msgs, images = [], []
        for fr in frames:
            img = Image.fromarray(fr) if isinstance(fr, np.ndarray) else fr
            images.append(img)
            msgs.append([{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": self.PROMPT.format(q=question)}]}])
        texts = [self.processor.apply_chat_template(m, tokenize=False,
                                                    add_generation_prompt=True)
                 for m in msgs]
        return self.processor(text=texts, images=images, padding=True,
                              return_tensors="pt").to(self.device)

    @torch.inference_mode()
    def score(self, frames: Sequence[np.ndarray], question: str) -> np.ndarray:
        """Return log P(Yes) - log P(No) per frame. Higher means more likely yes."""
        inputs = self._build_inputs(frames, question)
        logits = self.model(**inputs).logits[:, -1, :].float()
        logp = torch.log_softmax(logits, dim=-1)
        yes = torch.logsumexp(logp[:, self.yes_ids], dim=-1)
        no = torch.logsumexp(logp[:, self.no_ids], dim=-1)
        return (yes - no).cpu().numpy()

    @torch.inference_mode()
    def score_clips(self, clips: Sequence[Sequence[np.ndarray]], question: str
                    ) -> np.ndarray:
        """Same yes/no score, but each item is a short *clip* rather than a frame.

        A single frame cannot verify a query about something that happens over
        time -- "a man playing Connect Four with a woman, who wins the game" has
        no single frame that shows the winning. Passing the frames as a `video`
        block lets the model see the sequence, and costs roughly half the visual
        tokens of the same frames sent as separate images because Qwen2.5-VL
        merges adjacent frames temporally.
        """
        from PIL import Image

        msgs, videos = [], []
        for frames in clips:
            imgs = [Image.fromarray(f) if isinstance(f, np.ndarray) else f
                    for f in frames]
            videos.append(imgs)
            msgs.append([{"role": "user", "content": [
                {"type": "video"},
                {"type": "text", "text": self.PROMPT.format(q=question)}]}])
        texts = [self.processor.apply_chat_template(m, tokenize=False,
                                                    add_generation_prompt=True)
                 for m in msgs]
        inputs = self.processor(text=texts, videos=videos, padding=True,
                                return_tensors="pt").to(self.device)
        logits = self.model(**inputs).logits[:, -1, :].float()
        logp = torch.log_softmax(logits, dim=-1)
        yes = torch.logsumexp(logp[:, self.yes_ids], dim=-1)
        no = torch.logsumexp(logp[:, self.no_ids], dim=-1)
        return (yes - no).cpu().numpy()

    def visual_tokens_per_frame(self, hw: tuple[int, int]) -> int:
        """How many visual tokens a frame of this size actually costs."""
        h, w = hw
        px = h * w
        px = max(self.spec.min_pixels, min(self.spec.max_pixels, px))
        return int(px / (QWEN_PATCH * QWEN_PATCH) / (QWEN_MERGE * QWEN_MERGE))

    def describe(self) -> dict:
        return {"key": self.spec.key, "repo": self.spec.repo,
                "revision": self.spec.revision,
                "params_b": round(sum(p.numel() for p in self.model.parameters()) / 1e9, 2),
                "max_pixels": self.spec.max_pixels,
                "max_visual_tokens": self.spec.max_pixels // (QWEN_PATCH ** 2) // (QWEN_MERGE ** 2),
                "attn": self.attn_impl}


class QwenMultiFrameQA:
    """Answers a question given K selected frames, as the benchmark protocol wants.

    Frames are passed as a `video` content block rather than K separate images:
    that is how Qwen2.5-VL expects a frame sequence, it applies the temporal
    position encoding, and it costs roughly half the visual tokens of K images
    thanks to the temporal 2-frame merge.

    Shares weights with QwenYesNoScorer when constructed from one, so a single
    7B model serves both the refine stage and the benchmark.
    """

    def __init__(self, scorer: QwenYesNoScorer):
        self.model = scorer.model
        self.processor = scorer.processor
        self.tok = scorer.tok
        self.device = scorer.device
        self.spec = scorer.spec

    @torch.inference_mode()
    def answer(self, frames: Sequence[np.ndarray], prompt: str,
               max_new_tokens: int = 8) -> str:
        from PIL import Image
        imgs = [Image.fromarray(f) if isinstance(f, np.ndarray) else f for f in frames]
        msgs = [{"role": "user", "content": [
            {"type": "video"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(msgs, tokenize=False,
                                                  add_generation_prompt=True)
        inputs = self.processor(text=[text], videos=[imgs], padding=True,
                                return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                  do_sample=False,
                                  pad_token_id=self.tok.pad_token_id)
        gen = out[0, inputs["input_ids"].shape[1]:]
        return self.tok.decode(gen, skip_special_tokens=True).strip()

    @torch.inference_mode()
    def answer_letter_logits(self, frames: Sequence[np.ndarray], prompt: str,
                             letters: Sequence[str] = ("A", "B", "C", "D")) -> str:
        """Pick the option with the highest first-token probability.

        Deterministic, one forward pass, and immune to the model wandering off
        into prose -- which otherwise shows up as a parse failure and is scored
        as wrong, unfairly penalising whichever condition happens to trigger it.
        """
        from PIL import Image
        imgs = [Image.fromarray(f) if isinstance(f, np.ndarray) else f for f in frames]
        msgs = [{"role": "user", "content": [
            {"type": "video"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(msgs, tokenize=False,
                                                  add_generation_prompt=True)
        inputs = self.processor(text=[text], videos=[imgs], padding=True,
                                return_tensors="pt").to(self.device)
        logits = self.model(**inputs).logits[0, -1, :].float()
        ids = []
        for L in letters:
            cand = [self.tok.encode(f, add_special_tokens=False)
                    for f in (L, " " + L)]
            ids.append([c[0] for c in cand if c])
        scores = [torch.logsumexp(logits[torch.tensor(i, device=logits.device)], 0)
                  if i else torch.tensor(-1e9) for i in ids]
        return letters[int(torch.stack(scores).argmax())]


def _has_flash_attn() -> bool:
    try:
        import flash_attn  # noqa: F401
        return True
    except Exception:
        return False
