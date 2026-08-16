"""The most ordinary place of all: a cross-encoder over windows of a long document.

A production reranking stage almost always looks like this. The cross-encoder has
a 512-token limit, the documents do not, so you slide a window over the document,
score each window against the query, and take the best one. `max` over windows,
chosen because it is obvious.

The framework says that is correct exactly when one window answers the query. It
predicts a null on datasets whose documents answer in one place and a gain on
datasets that answer across several, so this is run on both kinds rather than on
whichever one is convenient.

Distinct from scripts/rag_pooling.py in the way that matters here: that used a
bi-encoder, embedding query and chunk separately. This is a cross-encoder, which
reads the query and the window together and is the thing people actually put in
a reranking stage. If the effect were an artifact of dot-product similarity it
would not survive the change.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from framesieve.pooling import topk_mean  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def windows_of(text: str, tok, width: int, stride: int) -> list[str]:
    """Token windows, decoded back to text so the cross-encoder does its own
    pairing. Overlap matters: a window boundary through the answer would
    manufacture exactly the fragmentation this experiment is measuring."""
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= width:
        return [text]
    out = []
    for s in range(0, max(1, len(ids) - width + stride), stride):
        out.append(tok.decode(ids[s:s + width]))
    return out or [text]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nfcorpus")
    ap.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--depth", type=int, default=50,
                    help="how many first-stage candidates to rerank per query")
    ap.add_argument("--width", type=int, default=160)
    ap.add_argument("--stride", type=int, default=80)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    out_path = args.out or f"runs/rerank_{args.dataset}.json"

    from late_interaction import load_beir
    from rag_pooling import Encoder, ndcg_at_k
    docs, qs, rel = load_beir(args.dataset)
    doc_ids = [i for i, _ in docs]
    print(f"{args.dataset}: {len(docs)} docs, {len(qs)} queries, "
          f"rerank depth {args.depth}")

    # First stage: a bi-encoder shortlist, because reranking every document with
    # a cross-encoder is not a thing anyone does and would take hours here.
    be = Encoder()
    Q = be(([t for _, t in qs]),
           prefix="Represent this sentence for searching relevant passages: ")
    D = be([t for _, t in docs])
    del be
    torch.cuda.empty_cache()
    shortlist = [np.argsort(-(D @ Q[i]))[:args.depth] for i in range(len(qs))]

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    ce = AutoModelForSequenceClassification.from_pretrained(
        args.model, torch_dtype=torch.float16).cuda().eval()

    KS = (1, 2, 3, 4, 6, 8)
    nd = {k: [] for k in KS}
    s1 = {k: [] for k in KS}
    nwin = []
    t0 = time.perf_counter()
    for qi, (qid, qtext) in enumerate(qs):
        cand = shortlist[qi]
        pairs, owner = [], []
        for j, di in enumerate(cand):
            ws = windows_of(docs[di][1], tok, args.width, args.stride)
            nwin.append(len(ws))
            pairs.extend([(qtext, w) for w in ws])
            owner.extend([j] * len(ws))
        scores = np.empty(len(pairs), dtype=np.float32)
        for b in range(0, len(pairs), args.batch):
            chunk = pairs[b:b + args.batch]
            enc = tok([p[0] for p in chunk], [p[1] for p in chunk],
                      padding=True, truncation=True, max_length=512,
                      return_tensors="pt").to("cuda")
            with torch.inference_mode():
                scores[b:b + len(chunk)] = ce(**enc).logits[:, 0].float().cpu().numpy()
        owner = np.asarray(owner)
        for k in KS:
            per_doc = np.array([topk_mean(scores[owner == j], k)
                                for j in range(len(cand))])
            order = np.argsort(-per_doc)[:10]
            ranked = [doc_ids[cand[j]] for j in order]
            nd[k].append(ndcg_at_k(ranked, rel[qid], 10))
            s1[k].append(float(rel[qid].get(ranked[0], 0) > 0))
        if (qi + 1) % 50 == 0:
            print(f"  {qi+1}/{len(qs)}  {time.perf_counter()-t0:.0f} s", flush=True)

    print(f"\n  windows per document: median {np.median(nwin):.0f}, "
          f"mean {np.mean(nwin):.1f}")
    print(f"  {'k':>4}{'nDCG@10':>10}{'success@1':>12}")
    rows = {}
    for k in KS:
        rows[k] = dict(ndcg10=100 * float(np.mean(nd[k])),
                       success1=100 * float(np.mean(s1[k])))
        print(f"  {k:>4}{rows[k]['ndcg10']:>10.2f}{rows[k]['success1']:>12.2f}")

    # paired bootstrap over queries against k=1, which is what `max` means here
    rng = np.random.default_rng(0)
    base_n = np.asarray(nd[1])
    base_s = np.asarray(s1[1])
    idx = rng.integers(0, len(base_n), size=(3000, len(base_n)))
    print(f"\n  {'k':>4}{'vs max, nDCG@10':>20}{'vs max, success@1':>22}")
    out_rows = []
    for k in KS:
        dn = np.asarray(nd[k]) - base_n
        ds = np.asarray(s1[k]) - base_s
        bn = np.percentile(dn[idx].mean(axis=1), [2.5, 97.5]) * 100
        bs = np.percentile(ds[idx].mean(axis=1), [2.5, 97.5]) * 100
        out_rows.append(dict(k=k, **rows[k],
                             d_ndcg=100 * float(dn.mean()), ndcg_ci=list(bn),
                             d_s1=100 * float(ds.mean()), s1_ci=list(bs)))
        if k == 1:
            print(f"  {k:>4}{'(baseline)':>20}{'':>22}")
            continue
        print(f"  {k:>4}{f'{100*dn.mean():+.2f} [{bn[0]:+.2f}, {bn[1]:+.2f}]':>20}"
              f"{f'{100*ds.mean():+.2f} [{bs[0]:+.2f}, {bs[1]:+.2f}]':>22}")

    sig = [r for r in out_rows if r["k"] > 1 and r["s1_ci"][0] > 0]
    print("\n  " + (f"max is beaten on success@1 at k = "
                    f"{[r['k'] for r in sig]}, interval excluding zero"
                    if sig else
                    "max is not beaten with a clean interval on this dataset"))

    with open(out_path, "w") as f:
        json.dump(dict(dataset=args.dataset, model=args.model, depth=args.depth,
                       width=args.width, stride=args.stride,
                       median_windows=float(np.median(nwin)), rows=out_rows),
                  f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
