"""Rerank BM25 top-k (from runs/bm25.jsonl) with the MedCPT cross-encoder; report MRR@10 per source.

Usage: python scripts/eval_rerank.py [--k 100]
"""
import argparse, json, time
from collections import defaultdict
from pathlib import Path

import duckdb, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=100)
    a = ap.parse_args()
    dev = {r["qid"]: r for r in map(json.loads, (ROOT / "dev/devset.jsonl").read_text().splitlines())}
    run = [json.loads(l) for l in (ROOT / "runs/bm25.jsonl").read_text().splitlines()]
    need = sorted({p for r in run for p in r["top"][: a.k]})
    t0 = time.time()
    txt = dict(duckdb.execute(f"select pmid, title || ' ' || abstract from '{ROOT}/data/parquet/*.parquet' where pmid in (select unnest(?))", [need]).fetchall())
    print(f"fetched {len(txt)} docs in {time.time()-t0:.0f}s", flush=True)
    devc = "mps"
    tok = AutoTokenizer.from_pretrained("ncbi/MedCPT-Cross-Encoder")
    m = AutoModelForSequenceClassification.from_pretrained("ncbi/MedCPT-Cross-Encoder", dtype=torch.float16).to(devc).eval()
    by = defaultdict(list); t0 = time.time()
    for r in run:
        q = dev[r["qid"]]; cand = r["top"][: a.k]
        scores = []
        with torch.no_grad():
            for s in range(0, len(cand), 50):
                pairs = [[q["query"], txt.get(p, "")] for p in cand[s:s + 50]]
                enc = tok(pairs, truncation=True, padding=True, max_length=512, return_tensors="pt").to(devc)
                scores += m(**enc).logits.squeeze(-1).float().cpu().tolist()
        ranked = [p for _, p in sorted(zip(scores, cand), reverse=True)]
        rank = ranked.index(q["pmid"]) + 1 if q["pmid"] in ranked else None
        for key in ("ALL", q["source"]):
            by[key].append(rank)
    dt = time.time() - t0
    out = {}
    for key in sorted(by, key=lambda k: (k != "ALL", k)):
        rs = by[key]
        out[key] = round(sum(1 / x for x in rs if x and x <= 10) / len(rs), 3)
        print(f"{key:12s} n={len(rs):4d} MRR@10 after rerank top{a.k}: {out[key]:.3f}")
    print(f"rerank {dt/len(run)*1000:.0f} ms/query")
    (ROOT / f"runs/rerank_top{a.k}.json").write_text(json.dumps({"mrr10": out, "ms_per_query": round(dt / len(run) * 1000)}, indent=1))
