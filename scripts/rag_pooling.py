"""The same knob in the most ordinary place: scoring a document by its chunks.

ColBERT is where this operation is most consequential, but it is not where most
people meet it. Most people meet it in a plain RAG pipeline: split documents into
passages, embed each passage with an off-the-shelf encoder, and score a document
by its best-matching passage. That last step is `max`, and it is chosen by
default rather than decided.

This runs that pipeline on BEIR with a standard sentence encoder and sweeps the
pooling depth, with two things done deliberately:

  head and average metrics side by side
      nDCG@10 averages over a ranked list; success@1 looks only at what came
      first. The video work found this effect is invisible under an averaged
      metric, so both are reported here rather than one. If the pattern repeats
      on text, that is a strong second instance of a claim about EVALUATION, not
      just about pooling.

  two chunkings
      sentences, and fixed windows of a few sentences with overlap, which is what
      production pipelines actually do. n per document differs a lot between
      them, and the account predicts the best k moves with the number of chunks
      that genuinely answer the query -- not with n itself.

Nothing here is late interaction and nothing here is video. If the curve looks
the same, the finding is about pooling.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from framesieve.pooling import topk_mean  # noqa: E402

SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'\[])")


def split_sentences(text: str, min_chars: int = 25) -> list[str]:
    """Cheap sentence split. Deliberately dependency-free: a proper segmenter
    would change the chunk boundaries slightly and none of the conclusions here
    turn on that, but an extra install would stop people trying it."""
    parts = [p.strip() for p in SENT.split(text) if p.strip()]
    out: list[str] = []
    for p in parts:
        # glue fragments onto the previous sentence rather than emitting chunks
        # too short to embed meaningfully
        if out and len(p) < min_chars:
            out[-1] = out[-1] + " " + p
        else:
            out.append(p)
    return out or [text.strip() or " "]


def windows(sents: list[str], size: int, stride: int) -> list[str]:
    if size <= 1:
        return sents
    return [" ".join(sents[i:i + size])
            for i in range(0, max(1, len(sents) - size + stride), stride)] or sents


class Encoder:
    def __init__(self, name="BAAI/bge-small-en-v1.5", device="cuda",
                 dtype=torch.float16, maxlen=512):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        self.m = AutoModel.from_pretrained(name, torch_dtype=dtype).to(device).eval()
        self.device, self.maxlen = device, maxlen

    def __call__(self, texts, batch=256, prefix=""):
        out = []
        for i in range(0, len(texts), batch):
            b = [prefix + t for t in texts[i:i + batch]]
            e = self.tok(b, padding=True, truncation=True, max_length=self.maxlen,
                         return_tensors="pt").to(self.device)
            with torch.inference_mode():
                h = self.m(**e).last_hidden_state[:, 0]     # bge uses the CLS token
                h = torch.nn.functional.normalize(h.float(), dim=-1)
            out.append(h.cpu().numpy().astype(np.float32))
        return np.concatenate(out)


def ndcg_at_k(ranked, rel: dict, k: int = 10) -> float:
    g = [rel.get(d, 0) for d in ranked[:k]]
    dcg = sum(v / np.log2(i + 2) for i, v in enumerate(g))
    idcg = sum(v / np.log2(i + 2)
               for i, v in enumerate(sorted(rel.values(), reverse=True)[:k]))
    return dcg / idcg if idcg > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scifact")
    ap.add_argument("--encoder", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--max-docs", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    out_path = args.out or f"runs/rag_{args.dataset}.json"

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from late_interaction import load_beir
    docs, qs, rel = load_beir(args.dataset)
    if args.max_docs:
        keep = {d for q in rel for d in rel[q]}
        docs = [d for d in docs if d[0] in keep][:args.max_docs] + \
               [d for d in docs if d[0] not in keep][:max(0, args.max_docs - len(keep))]
    print(f"{args.dataset}: {len(docs)} docs, {len(qs)} queries")

    enc = Encoder(args.encoder)
    # bge wants an instruction prefix on the query side and nothing on the
    # passage side; using the wrong one costs a couple of points and would be
    # charged to the pooling change if it were not handled here
    Q = enc([t for _, t in qs], prefix="Represent this sentence for searching "
                                       "relevant passages: ")
    doc_ids = [i for i, _ in docs]

    schemes = [("sentences", 1, 1), ("2-sentence windows, stride 1", 2, 1),
               ("4-sentence windows, stride 2", 4, 2)]
    KS = (1, 2, 3, 4, 6, 8, 12)
    results = {}

    for label, size, stride in schemes:
        t0 = time.perf_counter()
        chunks, owner = [], []
        for di, (_, text) in enumerate(docs):
            cs = windows(split_sentences(text), size, stride)
            chunks.extend(cs)
            owner.extend([di] * len(cs))
        owner = np.asarray(owner)
        C = enc(chunks)
        counts = np.bincount(owner, minlength=len(docs))
        print(f"\n{label}: {len(chunks):,} chunks, "
              f"median {np.median(counts):.0f} per doc, "
              f"encoded in {time.perf_counter()-t0:.0f} s")

        # chunk scores grouped by document, once; every k reads the same array
        order = np.argsort(owner, kind="stable")
        bounds = np.searchsorted(owner[order], np.arange(len(docs) + 1))
        Cg = torch.from_numpy(C[order]).cuda()
        Qt = torch.from_numpy(Q).cuda()

        rows = {}
        for k in KS:
            nd, s1 = [], []
            with torch.inference_mode():
                sims = (Qt @ Cg.T).cpu().numpy()          # [n_queries, n_chunks]
            for qi, (qid, _) in enumerate(qs):
                s = sims[qi]
                per_doc = np.array([
                    topk_mean(s[bounds[d]:bounds[d + 1]], k)
                    if bounds[d + 1] > bounds[d] else -1e9
                    for d in range(len(docs))])
                top = np.argsort(-per_doc)[:10]
                ranked = [doc_ids[j] for j in top]
                nd.append(ndcg_at_k(ranked, rel[qid], 10))
                s1.append(float(rel[qid].get(ranked[0], 0) > 0))
            rows[k] = dict(ndcg10=100 * float(np.mean(nd)),
                           success1=100 * float(np.mean(s1)))
            print(f"    k={k:<3} nDCG@10 {rows[k]['ndcg10']:>6.2f}   "
                  f"success@1 {rows[k]['success1']:>6.2f}", flush=True)

        bn = max(rows, key=lambda k: rows[k]["ndcg10"])
        bs = max(rows, key=lambda k: rows[k]["success1"])
        results[label] = dict(rows={str(k): v for k, v in rows.items()},
                              best_ndcg_k=bn, best_success_k=bs,
                              median_chunks=float(np.median(counts)))
        print(f"    best k by nDCG@10  : {bn}  "
              f"({rows[bn]['ndcg10']-rows[1]['ndcg10']:+.2f} over max)")
        print(f"    best k by success@1: {bs}  "
              f"({rows[bs]['success1']-rows[1]['success1']:+.2f} over max)")
        del Cg

    print("\nsummary -- does the head metric move the answer?")
    print(f"  {'chunking':<30}{'chunks/doc':>12}{'best k, nDCG':>14}"
          f"{'best k, success@1':>19}")
    for label, r in results.items():
        print(f"  {label:<30}{r['median_chunks']:>12.0f}{r['best_ndcg_k']:>14}"
              f"{r['best_success_k']:>19}")

    with open(out_path, "w") as f:
        json.dump(dict(dataset=args.dataset, encoder=args.encoder,
                       n_docs=len(docs), n_queries=len(qs), schemes=results),
                  f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
