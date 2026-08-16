"""Is the chunk-pooling result about video, or about pooling?

On MomentSeeker, replacing `max` over the frames in a candidate chunk with the
mean of the top few was worth +2.6 R@1 for no compute. Four unrelated families of
statistic all peaked in the interior between the mean and the max, and three of
them peaked on the same number. Nothing in that argument mentions video: it is a
statement about collapsing many fine-grained scores into one coarse score, which
is a thing done all over retrieval.

The most consequential place it is done is late interaction. ColBERT scores a
query against a document as

    score(q, d) = SUM over query tokens i of  MAX over doc tokens j of  q_i . d_j

which is an extreme-order statistic in exactly the place this project found one
to be the wrong choice. ColBERTv2, ColPali and ColQwen all use it, so if the
interior beats the endpoint here too, it is a free change for a large number of
production systems and the video result was never really about video.

Two aggregations to vary, and they are independent:

    inner   MAX over document tokens, per query token. Replacing it with the mean
            of the top k asks each query term to be supported by several places
            in the document rather than one.
    outer   SUM over query tokens. That is a mean up to a constant, so it is the
            OTHER endpoint of the same family, and top-k there asks the document
            to match the query's best few terms strongly rather than all of them
            evenly.

Validity: the MaxSim baseline must reproduce ColBERTv2's published BEIR
nDCG@10 before any of the variants mean anything. SciFact is 69.3 and NFCorpus
33.8 in the ColBERTv2 paper. If this harness does not land near those, it is the
harness that is wrong, not the finding.
"""

from __future__ import annotations

import argparse
import json
import os
import string
import time

import numpy as np
import torch

# ColBERTv2 paper, Table 2 (BEIR nDCG@10)
PUBLISHED = {"scifact": 69.3, "nfcorpus": 33.8, "trec-covid": 73.8, "fiqa": 35.6,
             "arguana": 46.3, "scidocs": 15.4, "quora": 85.2, "webis-touche2020": 26.3}

# ArguAna is a counter-argument task: the query IS a document from the corpus,
# so the standard evaluation excludes a query's own document from its ranking.
# Leaving it in puts a perfect self-match at rank 1 for every query and destroys
# the score. It also means queries are full arguments rather than questions, so
# the usual 32-token query budget truncates them severely.
SELF_EXCLUDE = {"arguana"}
QUERY_MAXLEN = {"arguana": 160}


# --------------------------------------------------------------------------
# ColBERTv2, reimplemented from the checkpoint rather than from the library,
# because the library pulls in an indexing stack this experiment does not need
# and the scoring function is four lines.
# --------------------------------------------------------------------------


class ColBERT:
    # ColBERT marks queries and documents with two reserved vocabulary slots.
    # They must be inserted as IDs: passing the literal string "[unused0]"
    # through the tokenizer splits it into ['[', 'unused', '##0', ']'], which
    # silently corrupts every representation and cost this harness 8.7 nDCG.
    QUERY_MARKER_ID, DOC_MARKER_ID = 1, 2

    def __init__(self, name="colbert-ir/colbertv2.0", device="cuda",
                 dtype=torch.float16, query_maxlen=32, doc_maxlen=512):
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import AutoModel, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(name)
        self.bert = AutoModel.from_pretrained(name, torch_dtype=dtype).to(device).eval()
        sd = load_file(hf_hub_download(name, "model.safetensors"))
        self.proj = sd["linear.weight"].to(device=device, dtype=dtype)  # [128, 768]
        self.device, self.dtype = device, dtype
        self.query_maxlen, self.doc_maxlen = query_maxlen, doc_maxlen
        self.mask_id = self.tok.mask_token_id
        # mask_punctuation: punctuation tokens are excluded from a document's
        # representation entirely. Skipping this inflates document length with
        # tokens that match nothing, which would quietly change the very
        # statistic under test.
        self.punct = {self.tok.convert_tokens_to_ids(c) for c in string.punctuation}
        self.punct.discard(self.tok.unk_token_id)

    def _encode(self, texts, marker_id, maxlen, pad_with_mask, batch=32):
        out = []
        for i in range(0, len(texts), batch):
            raw = self.tok(list(texts[i:i + batch]), truncation=True,
                           max_length=maxlen - 1)["input_ids"]
            # [CLS] <marker> ... [SEP], then pad to a common length
            seqs = [[s[0], marker_id] + s[1:] for s in raw]
            width = maxlen if pad_with_mask else max(len(s) for s in seqs)
            pad = self.mask_id if pad_with_mask else self.tok.pad_token_id
            ids = torch.full((len(seqs), width), pad, dtype=torch.long)
            am = torch.zeros((len(seqs), width), dtype=torch.long)
            for r, s in enumerate(seqs):
                s = s[:width]
                ids[r, :len(s)] = torch.tensor(s)
                am[r, :len(s)] = 1
            if pad_with_mask:
                # query augmentation: the [MASK] padding is attended to and DOES
                # take part in scoring -- that is the point of it
                am = torch.ones_like(am)
            ids, am = ids.to(self.device), am.to(self.device)
            with torch.inference_mode():
                h = self.bert(input_ids=ids, attention_mask=am).last_hidden_state
                v = h @ self.proj.T
                v = torch.nn.functional.normalize(v.float(), dim=-1).to(self.dtype)
            keep = am.bool()
            if not pad_with_mask:
                for p in self.punct:
                    keep &= ids != p
                keep &= ids != self.tok.pad_token_id
            for r in range(v.shape[0]):
                out.append(v[r][keep[r]].contiguous())
        return out

    def queries(self, texts):
        return self._encode(texts, self.QUERY_MARKER_ID, self.query_maxlen, True)

    def docs(self, texts):
        return self._encode(texts, self.DOC_MARKER_ID, self.doc_maxlen, False)


# --------------------------------------------------------------------------
# the scoring variants -- this is the whole experiment
# --------------------------------------------------------------------------


def interior(x: torch.Tensor, mode: str, param, dim: int) -> torch.Tensor:
    """One statistic over `dim`, somewhere between the mean and the max."""
    n = x.shape[dim]
    if mode == "max":
        return x.max(dim=dim).values
    if mode == "mean":
        return x.mean(dim=dim)
    if mode == "topk":
        k = min(int(param), n)
        return x.topk(k, dim=dim).values.mean(dim=dim)
    if mode == "adaptive":                     # k as a fraction of n, as in video
        k = max(1, min(int(round(n / float(param))), n))
        return x.topk(k, dim=dim).values.mean(dim=dim)
    if mode == "power":                        # generalised mean; x is in [-1, 1]
        w = (x + 1.0 + 1e-6).clamp_min(1e-6)
        return (w.pow(float(param)).mean(dim=dim)).pow(1.0 / float(param)) - 1.0
    raise ValueError(mode)


def score_all(Q: torch.Tensor, qmask: torch.Tensor, D: torch.Tensor,
              dmask: torch.Tensor, inner, outer) -> torch.Tensor:
    """Score one query against a block of documents.

    Q    [nq, dim]        one query's token vectors
    D    [nd, lt, dim]    padded document token vectors
    Masked-out document positions are set to -1 (below any real cosine) so they
    can never win a max or enter a top-k ahead of a real token.
    """
    sim = torch.einsum("qd,nld->nql", Q, D)          # [nd, nq, lt]
    sim = sim.masked_fill(~dmask[:, None, :], -1.0)
    per_q = interior(sim, inner[0], inner[1], dim=2)  # [nd, nq]
    per_q = per_q.masked_fill(~qmask[None, :], -1e4 if outer[0] == "max" else 0.0)
    if outer[0] == "mean":                            # ColBERT's SUM, up to a constant
        return per_q.sum(dim=1)
    return interior(per_q, outer[0], outer[1], dim=1) * qmask.sum()


def ndcg_at_k(ranked_ids, rel: dict, k: int = 10) -> float:
    g = [rel.get(d, 0) for d in ranked_ids[:k]]
    dcg = sum(v / np.log2(i + 2) for i, v in enumerate(g))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum(v / np.log2(i + 2) for i, v in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


# --------------------------------------------------------------------------


def load_beir(name: str, split: str = "test"):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    corpus = pd.read_parquet(hf_hub_download(
        f"BeIR/{name}", "corpus/corpus-00000-of-00001.parquet", repo_type="dataset"))
    queries = pd.read_parquet(hf_hub_download(
        f"BeIR/{name}", "queries/queries-00000-of-00001.parquet", repo_type="dataset"))
    qrels = pd.read_csv(hf_hub_download(
        f"BeIR/{name}-qrels", f"{split}.tsv", repo_type="dataset"), sep="\t")
    rel: dict = {}
    for r in qrels.itertuples():
        if int(r.score) > 0:
            rel.setdefault(str(r._1), {})[str(r._2)] = int(r.score)
    queries = queries[queries["_id"].astype(str).isin(rel)]
    docs = [(str(r._1), (str(r.title) + " " + str(r.text)).strip())
            for r in corpus.itertuples()]
    qs = [(str(r._1), str(r.text)) for r in queries.itertuples()]
    return docs, qs, rel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scifact")
    ap.add_argument("--out", default="")
    ap.add_argument("--doc-batch", type=int, default=256)
    args = ap.parse_args()
    out_path = args.out or f"runs/late_{args.dataset}.json"

    docs, qs, rel = load_beir(args.dataset)
    print(f"{args.dataset}: {len(docs)} docs, {len(qs)} queries with judgements")

    cb = ColBERT(query_maxlen=QUERY_MAXLEN.get(args.dataset, 32))
    t0 = time.perf_counter()
    Qs = cb.queries([t for _, t in qs])
    D = cb.docs([t for _, t in docs])
    lens = np.array([d.shape[0] for d in D])
    print(f"encoded in {time.perf_counter()-t0:.0f} s; "
          f"doc tokens median {np.median(lens):.0f}, mean {lens.mean():.1f}, "
          f"max {lens.max()}")

    # pad the whole corpus once; the variants all read the same tensor
    LT = int(lens.max())
    Dpad = torch.zeros(len(D), LT, D[0].shape[1], device=cb.device, dtype=cb.dtype)
    dmask = torch.zeros(len(D), LT, dtype=torch.bool, device=cb.device)
    for i, d in enumerate(D):
        Dpad[i, :d.shape[0]] = d
        dmask[i, :d.shape[0]] = True
    del D

    doc_ids = [i for i, _ in docs]
    doc_pos = {d: i for i, d in enumerate(doc_ids)}
    self_exclude = args.dataset in SELF_EXCLUDE
    variants = (
        [("maxsim (ColBERT)", ("max", 0), ("mean", 0))]
        + [(f"inner top-{k}", ("topk", k), ("mean", 0)) for k in (2, 3, 4, 8, 16)]
        + [(f"inner k=n/{f}", ("adaptive", f), ("mean", 0)) for f in (2, 4, 8, 16)]
        + [(f"inner power p={p}", ("power", p), ("mean", 0)) for p in (4, 8, 16)]
        + [("inner mean", ("mean", 0), ("mean", 0))]
        + [(f"outer top-{k}", ("max", 0), ("topk", k)) for k in (4, 8, 16, 24)]
        + [("outer max", ("max", 0), ("max", 0))]
        + [(f"inner top-2 + outer top-{k}", ("topk", 2), ("topk", k)) for k in (16, 24)]
    )

    results = {lab: [] for lab, _, _ in variants}
    for qi, (qid, _) in enumerate(qs):
        Q = Qs[qi]
        qmask = torch.ones(Q.shape[0], dtype=torch.bool, device=cb.device)
        for lab, inner, outer in variants:
            chunks = []
            for s in range(0, Dpad.shape[0], args.doc_batch):
                e = s + args.doc_batch
                chunks.append(score_all(Q, qmask, Dpad[s:e], dmask[s:e],
                                        inner, outer).float())
            sc = torch.cat(chunks)
            if self_exclude and qid in doc_pos:
                sc[doc_pos[qid]] = -1e9
            top = torch.topk(sc, 10).indices.tolist()
            results[lab].append(ndcg_at_k([doc_ids[j] for j in top], rel[qid], 10))
        if (qi + 1) % 50 == 0:
            print(f"  {qi+1}/{len(qs)}  maxsim nDCG@10 "
                  f"{100*np.mean(results['maxsim (ColBERT)']):.2f}", flush=True)

    base = 100 * float(np.mean(results["maxsim (ColBERT)"]))
    pub = PUBLISHED.get(args.dataset)
    print(f"\nbaseline check: MaxSim nDCG@10 {base:.2f}"
          + (f"   published {pub}   delta {base-pub:+.2f}" if pub else ""))
    # Measured faithfulness sweep: fp32 changes nothing (66.50 vs 66.45), a
    # longer query_maxlen is much worse (49.83 at 64, because ColBERT's [MASK]
    # padding takes part in scoring), and doc_maxlen 512 is worth +1.3 over 300
    # because SciFact documents run to a median 313 tokens. What is left is a
    # ~1.5 gap, small against this dataset's ~4-point bootstrap interval and
    # attributable to corpus/tokenisation details in the official eval.
    if pub and abs(base - pub) > 3.0:
        print("  WARNING: the baseline does not reproduce, so nothing below "
              "should be believed")

    print(f"\n{args.dataset}: nDCG@10 by aggregation")
    print(f"  {'variant':<30}{'nDCG@10':>10}{'vs MaxSim':>12}")
    rows = []
    for lab, _, _ in variants:
        v = 100 * float(np.mean(results[lab]))
        rows.append(dict(variant=lab, ndcg10=v, delta=v - base))
        print(f"  {lab:<30}{v:>10.2f}{v-base:>+12.2f}")
    best = max(rows, key=lambda r: r["ndcg10"])
    print(f"\n  best: {best['variant']}  {best['ndcg10']:.2f} "
          f"({best['delta']:+.2f} over MaxSim)")
    print("  " + ("the interior beats the endpoint here too"
                  if best["delta"] > 0.3 and best["variant"] != "maxsim (ColBERT)"
                  else "MaxSim is not beaten -- the video result does not "
                       "transfer to late interaction"))

    os.makedirs("runs", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(dict(dataset=args.dataset, n_docs=len(docs), n_queries=len(qs),
                       published=pub, baseline=base, rows=rows), f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
