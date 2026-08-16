"""The same dial, mirrored: aggregating step rewards in a process reward model.

Everywhere else in this project the pooling is over things that might MATCH, and
the question is how many of them should. A process reward model inverts it. It
emits a reward per reasoning step, and the standard way to turn those into one
score for a solution is `min` -- a chain is only as good as its worst link.

That is the same family seen from the other end:

    min(x)  == -topk_mean(-x, k=1)      "one bad step is fatal"
    mean(x) == -topk_mean(-x, k=n)      "every step counts equally"

and the account this post defends says the right k tracks how many of the n steps
are actually decisive. If a wrong solution typically goes wrong in exactly one
place, `min` is correct and moving off it should cost. If wrongness is spread
across several steps -- which is what it looks like when a model loses the plot
halfway through -- the interior should win, and the field's default is leaving
accuracy behind.

Setup, with no generation required. Math-Shepherd ships multiple candidate
solutions per GSM8K problem with every step hand-labelled + or -, so:

    the cheap per-step score   Qwen2.5-Math-PRM-7B's reward at each step
    the label                  Math-Shepherd's step tags; a solution is correct
                               iff every step is +
    m, measured                the number of MIS-labelled steps in a wrong
                               solution, which is the mirrored quantity

The task is best-of-n: rank a problem's candidate solutions by the aggregated
reward and check whether the top-ranked one is fully correct. That is a head
metric by construction, which the metric experiment says is the only kind that
can see this.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

STEP_RE = re.compile(r"Step \d+:")


def parse_row(row: dict):
    """Math-Shepherd encodes step labels by replacing a trailing tag token.

    `input` holds the problem and the solution with a placeholder after each
    step; `label` is the same string with the placeholder replaced by + or -.
    Diffing them recovers a per-step label without guessing at the tag token.
    """
    inp, lab = row["input"], row["label"]
    parts = STEP_RE.split(inp)
    if len(parts) < 2:
        return None
    problem = parts[0].strip()
    # rebuild the steps with their markers, then read the tag off `label`
    starts = [m.start() for m in STEP_RE.finditer(inp)] + [len(inp)]
    steps, tags = [], []
    for a, b in zip(starts[:-1], starts[1:]):
        seg_in = inp[a:b]
        seg_lab = lab[a:b] if b <= len(lab) else lab[a:]
        # the two strings differ only in the tag character at the very end
        t = None
        for ch in reversed(seg_lab.strip()):
            if ch in "+-":
                t = ch
                break
        if t is None:
            return None
        # strip the placeholder token from the step text
        text = seg_in.strip()
        text = re.sub(r"\s*[^\s]*$", "", text) if text.endswith(("ки", "+", "-")) \
            else text
        steps.append(text.strip())
        tags.append(t)
    if not steps or len(steps) != len(tags):
        return None
    return dict(problem=problem, steps=steps, tags=tags,
                correct=all(t == "+" for t in tags),
                n_bad=sum(t == "-" for t in tags))


def load(limit_problems: int, min_solutions: int = 2):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("peiyi9979/Math-Shepherd", "math-shepherd.jsonl",
                           repo_type="dataset")
    by_problem: dict = collections.OrderedDict()
    with open(path) as f:
        for line in f:
            r = parse_row(json.loads(line))
            if r is None:
                continue
            by_problem.setdefault(r["problem"], []).append(r)
    # a best-of-n comparison needs both a correct and an incorrect candidate,
    # otherwise the ranking cannot be right or wrong
    keep = [(p, sols) for p, sols in by_problem.items()
            if len(sols) >= min_solutions
            and any(s["correct"] for s in sols)
            and any(not s["correct"] for s in sols)]
    return keep[:limit_problems]


class PRM:
    """Qwen2.5-Math-PRM-7B, assembled from the checkpoint rather than its
    `trust_remote_code` module.

    The shipped modeling file targets an older transformers and dies on
    `config.pad_token_id` under 5.x. The model is not exotic -- a Qwen2 backbone
    plus a two-layer MLP head that classifies each `<extra_0>` position -- so the
    backbone comes from transformers and only the head is lifted out of the
    checkpoint. Structure verified against the weight index: score.0 and score.2
    with a ReLU between them.
    """

    def __init__(self, name="Qwen/Qwen2.5-Math-PRM-7B", device="cuda"):
        import json as _json

        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import AutoTokenizer, Qwen2Model

        self.tok = AutoTokenizer.from_pretrained(name)
        self.m = Qwen2Model.from_pretrained(name, torch_dtype=torch.bfloat16
                                            ).to(device).eval()
        idx = _json.load(open(hf_hub_download(name, "model.safetensors.index.json")))
        want = {k: v for k, v in idx["weight_map"].items() if k.startswith("score.")}
        head = {}
        for shard in sorted(set(want.values())):
            sd = load_file(hf_hub_download(name, shard))
            head.update({k: sd[k] for k in want if k in sd})
        h = self.m.config.hidden_size
        self.head = torch.nn.Sequential(
            torch.nn.Linear(h, head["score.0.weight"].shape[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(head["score.0.weight"].shape[0], 2),
        ).to(device=device, dtype=torch.bfloat16).eval()
        with torch.no_grad():
            self.head[0].weight.copy_(head["score.0.weight"])
            self.head[0].bias.copy_(head["score.0.bias"])
            self.head[2].weight.copy_(head["score.2.weight"])
            self.head[2].bias.copy_(head["score.2.bias"])
        self.sep_id = self.tok.convert_tokens_to_ids("<extra_0>")
        assert self.sep_id is not None and self.sep_id >= 0, "no <extra_0> token"
        self.device = device

    def score(self, problem: str, steps: list[str]) -> list[float]:
        """One reward per step, as the model card prescribes."""
        msgs = [
            {"role": "system", "content":
             "Please reason step by step, and put your final answer within "
             "\\boxed{}."},
            {"role": "user", "content": problem},
            {"role": "assistant", "content": "<extra_0>".join(steps) + "<extra_0>"},
        ]
        text = self.tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=False)
        ids = self.tok(text, return_tensors="pt").input_ids.to(self.device)
        with torch.inference_mode():
            h = self.m(input_ids=ids).last_hidden_state
            logits = self.head(h)
        mask = (ids == self.sep_id)
        probs = F.softmax(logits.float(), dim=-1) * mask.unsqueeze(-1)
        pos = probs[0][probs[0] != 0].view(-1, 2)[:, 1]
        return pos.cpu().tolist()


class GenerativePRM:
    """RLHFlow's Llama-3.1-8B PRM, a deliberately different second family.

    Qwen's PRM is a classifier: a two-layer head over the backbone reads each
    `<extra_0>` position. This one is generative: the solution is formatted as a
    conversation where the assistant answers "+" after every step, and the step
    reward is the model's probability of "+" against "-" at that position. Same
    task, entirely different mechanism -- and trained on Deepseek-generated
    annotations rather than on Math-Shepherd, so it is not scoring its own
    training data.
    """

    def __init__(self, name="RLHFlow/Llama3.1-8B-PRM-Deepseek-Data",
                 device="cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        self.m = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.bfloat16).to(device).eval()
        self.plus = self.tok.encode("+")[-1]
        self.minus = self.tok.encode("-")[-1]
        self.device = device

    def score(self, problem: str, steps: list[str]) -> list[float]:
        conv = []
        for i, st in enumerate(steps):
            conv.append({"role": "user",
                         "content": (problem + " " + st) if i == 0 else st})
            conv.append({"role": "assistant", "content": "+"})
        # transformers 5.x returns a BatchEncoding here where 4.x returned a
        # bare tensor; accept either rather than pinning a version
        enc = self.tok.apply_chat_template(conv, return_tensors="pt")
        ids = (enc["input_ids"] if hasattr(enc, "keys") else enc).to(self.device)
        with torch.inference_mode():
            logits = self.m(ids).logits[0][:, [self.plus, self.minus]]
        probs = logits.float().softmax(dim=-1)[:, 0]
        # The reward for a step is read at the assistant "+" it produced. Masking
        # on the token id alone over-collects: arithmetic in the step text ("3+5")
        # contains the same token. So find where each assistant turn ENDS by
        # rendering the conversation prefix, then take the last "+" before that
        # point, which can only be the one the assistant emitted.
        pos = []
        for i in range(len(steps)):
            pre = self.tok.apply_chat_template(conv[: 2 * i + 2],
                                               return_tensors="pt")
            pre = pre["input_ids"] if hasattr(pre, "keys") else pre
            L = min(int(pre.shape[-1]), int(ids.shape[-1]))
            j = L - 1
            while j >= 0 and int(ids[0, j]) != self.plus:
                j -= 1
            if j >= 0:
                pos.append(j)
        if len(pos) != len(steps):
            return []
        return probs[torch.tensor(pos, device=probs.device)].cpu().tolist()


def bottomk_mean(x: np.ndarray, k: int) -> float:
    """Mirror of topk_mean: the mean of the k LOWEST step rewards.

    k = 1 is min, the field's default; k = n is the mean.
    """
    k = int(max(1, min(k, len(x))))
    return float(np.sort(x)[:k].mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", type=int, default=400)
    ap.add_argument("--min-solutions", type=int, default=2)
    ap.add_argument("--prm", default="qwen", choices=["qwen", "rlhflow"],
                    help="which reward-model family; two of them, because one "
                         "is a single point of evidence")
    ap.add_argument("--out", default="runs/prm_pooling.json")
    args = ap.parse_args()

    data = load(args.problems, args.min_solutions)
    n_sol = sum(len(s) for _, s in data)
    print(f"{len(data)} problems with both a correct and an incorrect candidate, "
          f"{n_sol} solutions")
    steps_per = [len(s["steps"]) for _, sols in data for s in sols]
    print(f"steps per solution: median {np.median(steps_per):.0f}, "
          f"mean {np.mean(steps_per):.1f}")
    bad = [s["n_bad"] for _, sols in data for s in sols if not s["correct"]]
    print(f"mis-labelled steps in a WRONG solution: median {np.median(bad):.0f}, "
          f"mean {np.mean(bad):.2f}  <- this is the mirrored m")

    prm = PRM() if args.prm == "qwen" else GenerativePRM()
    scored = []
    step_r, step_y = [], []
    for i, (problem, sols) in enumerate(data):
        rec = []
        for s in sols:
            r = prm.score(problem, s["steps"])
            if len(r) != len(s["steps"]):
                continue
            step_r.extend(r)
            step_y.extend([t == "+" for t in s["tags"]])
            rec.append(dict(rewards=r, correct=s["correct"], n_bad=s["n_bad"],
                            n_steps=len(s["steps"])))
        if len(rec) >= 2 and any(x["correct"] for x in rec) \
                and any(not x["correct"] for x in rec):
            scored.append(rec)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(data)} problems", flush=True)
    print(f"{len(scored)} problems usable after scoring")

    # Validity check, the same discipline as the ColBERT baseline: if the PRM
    # cannot tell a good step from a bad one, no aggregation of its output means
    # anything. Reconstructing the head from the checkpoint is exactly the kind
    # of thing that fails silently.
    x = np.asarray(step_r, dtype=float)
    y = np.asarray(step_y, dtype=bool)
    rk = np.argsort(np.argsort(x)) + 1
    a, b = int(y.sum()), int((~y).sum())
    auc = (rk[y].sum() - a * (a + 1) / 2) / (a * b) if a and b else float("nan")
    print(f"  PRM step reward vs step label: AUC {auc:.3f} "
          f"over {len(x):,} steps ({a:,} good, {b:,} bad)")
    print(f"  median reward on a good step {np.median(x[y]):.3f}, "
          f"on a bad step {np.median(x[~y]):.3f}")
    # Why min's cost differs so much between reward models: `min` is maximally
    # sensitive to a single low score, so what matters is not how well the model
    # RANKS steps (AUC) but how often a fully correct solution still contains one
    # step the model scores low. That is directly measurable.
    good_min = [float(np.min(x["rewards"])) for rec in scored for x in rec
                if x["correct"]]
    bad_min = [float(np.min(x["rewards"])) for rec in scored for x in rec
               if not x["correct"]]
    if good_min and bad_min:
        gm, bm = np.asarray(good_min), np.asarray(bad_min)
        print(f"  min-over-steps of a CORRECT solution: median {np.median(gm):.3f}, "
              f"10th pct {np.percentile(gm, 10):.3f}")
        print(f"  min-over-steps of a WRONG solution:   median {np.median(bm):.3f}, "
              f"90th pct {np.percentile(bm, 90):.3f}")
        print(f"  overlap: {100*np.mean(gm < np.median(bm)):.1f}% of correct "
              f"solutions have a minimum below the median wrong one")

    if auc < 0.7:
        print("  WARNING: the reconstructed PRM barely discriminates -- nothing "
              "below should be believed")

    KS = (1, 2, 3, 4, 6, 8)
    variants = ([("min (the default)", "bottomk", 1)]
                + [(f"mean of bottom {k}", "bottomk", k) for k in KS[1:]]
                + [("mean of all steps", "mean", 0),
                   ("last step only", "last", 0),
                   ("product", "prod", 0)])

    def hits_for(mode, k, recs):
        out = []
        for rec in recs:
            sc = []
            for x in rec:
                r = np.asarray(x["rewards"], dtype=float)
                if mode == "bottomk":
                    sc.append(bottomk_mean(r, k))
                elif mode == "mean":
                    sc.append(float(r.mean()))
                elif mode == "last":
                    sc.append(float(r[-1]))
                else:
                    sc.append(float(np.prod(r)))
            out.append(float(rec[int(np.argmax(sc))]["correct"]))
        return np.asarray(out)

    print("\nbest-of-n accuracy: is the top-ranked solution fully correct?")
    print(f"  {'aggregation':<22}{'accuracy':>10}{'vs min':>9}{'95% CI, paired':>20}")
    rng = np.random.default_rng(0)
    H = {}
    for lab, mode, k in variants:
        H[lab] = hits_for(mode, k, scored)
    base_h = H["min (the default)"]
    # The comparison is paired -- same problems, same candidates, only the
    # aggregation differs -- so bootstrap the DIFFERENCE rather than the two
    # accuracies separately, which would be a much wider and wrong interval.
    idx = rng.integers(0, len(base_h), size=(4000, len(base_h)))
    rows = []
    for lab, mode, k in variants:
        d = H[lab] - base_h
        boot = d[idx].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append(dict(variant=lab, mode=mode, k=k, acc=100 * H[lab].mean(),
                         delta=100 * d.mean(), lo=100 * lo, hi=100 * hi))
        ci = "" if lab.startswith("min") else f"[{100*lo:+.2f}, {100*hi:+.2f}]"
        print(f"  {lab:<22}{100*H[lab].mean():>10.2f}{100*d.mean():>+9.2f}{ci:>20}")

    best = max(rows, key=lambda r: r["acc"])
    sig = [r for r in rows if r["lo"] > 0]
    print(f"\n  best: {best['variant']}  {best['acc']:.2f} "
          f"({best['delta']:+.2f} over min, 95% CI "
          f"[{best['lo']:+.2f}, {best['hi']:+.2f}])")
    print(f"  aggregations beating min with an interval excluding zero: "
          f"{len(sig)}/{len(rows)-1}")
    print("  " + ("min is beaten, so the field's default step aggregation is "
                  "leaving accuracy behind"
                  if sig else
                  "no variant beats min with a clean interval -- consistent with "
                  "wrongness being\n  concentrated in one step, which is what the "
                  "mirrored account would predict"))

    # Does the optimum track the mirrored m -- how many steps are actually wrong?
    print("\n  does best k track the number of mis-labelled steps?")
    print(f"  {'wrong steps in this problem':<30}{'n':>5}"
          + "".join(f"{'k='+str(k):>8}" for k in KS) + f"{'best k':>9}")
    strata = []
    for lab, lo_, hi_ in (("1", 1, 1), ("2", 2, 2), ("3-4", 3, 4), ("5+", 5, 99)):
        sub = [rec for rec in scored
               if lo_ <= np.median([x["n_bad"] for x in rec
                                    if not x["correct"]]) <= hi_]
        if len(sub) < 25:
            continue
        acc = {k: float(hits_for("bottomk", k, sub).mean()) for k in KS}
        bk = max(acc, key=lambda k: acc[k])
        strata.append(dict(group=lab, n=len(sub), acc=acc, best_k=bk))
        print(f"  {lab:<30}{len(sub):>5}"
              + "".join(f"{100*acc[k]:>8.1f}" for k in KS) + f"{bk:>9}")
    if len(strata) >= 3:
        mid = [float(np.mean([int(v) for v in st["group"].replace("+", "")
                              .split("-")])) for st in strata]
        bk = [st["best_k"] for st in strata]
        c = (float(np.corrcoef(np.log(mid), np.log(bk))[0, 1])
             if np.std(bk) > 0 else 0.0)
        print(f"  mirrored m: {mid}   best k: {bk}   correlation {c:+.3f}")

    with open(args.out, "w") as f:
        json.dump(dict(n_problems=len(scored), n_solutions=n_sol,
                       median_bad_steps=float(np.median(bad)),
                       mean_bad_steps=float(np.mean(bad)),
                       step_auc=float(auc),
                       good_min_median=float(np.median(good_min)) if good_min else None,
                       bad_min_median=float(np.median(bad_min)) if bad_min else None,
                       overlap=float(np.mean(np.asarray(good_min)
                                             < np.median(bad_min)))
                       if good_min and bad_min else None,
                       rows=rows,
                       strata=strata if "strata" in dir() else []), f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
