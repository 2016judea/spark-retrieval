# spark-retrieval

Known-item retrieval over the frozen PubMed baseline, built for the NIH SPARK PubMed/PMC
Challenge, Task 1. Someone describes a paper they remember ("landmark paper demonstrating
CRISPR-Cas9 RNA-programmable genome editing in vitro") and the system returns its PMID,
scored by MRR.

Rules this repo is built to: open-weight models and open-source software only, everything
ships in a Docker image, no network at evaluation, corpus = the SPARK PubMed baseline
(1,306 files, 1959-01-01 to 2026-07-31).

## Status (2026-10-21)

| stage | MRR@10 on the proxy dev set (727 queries) |
|---|---|
| BM25 | 0.414 |
| BM25 top-100 + MedCPT cross-encoder | 0.500 |

Dense retrieval, fusion and Docker are next. Measured costs are in [PLAN.md](PLAN.md).

## Run it

```bash
uv venv -p 3.12 .venv && uv pip install -p .venv pyarrow lxml tantivy polars duckdb torch transformers numpy faiss-cpu huggingface_hub
caffeinate -i -s scripts/download_baseline.sh      # 37 min, 54 GB, md5-verified
.venv/bin/python scripts/parse_baseline.py         # ~9 min, 21 GB parquet
.venv/bin/python scripts/build_bm25.py             # 6 min, 16 GB tantivy index
.venv/bin/python scripts/build_devset.py           # needs ollama + gemma3:4b
.venv/bin/python scripts/eval.py                   # BM25 MRR@10 / recall
.venv/bin/python scripts/eval_rerank.py --k 100    # + cross-encoder
.venv/bin/python scripts/bench_dense.py            # encoder / reranker timings
```

## Dev set (offline only, never shipped in the system)

`dev/devset.jsonl`, 727 queries, each with one gold PMID:
- **public** (200): BioASQ questions with exactly one relevant PMID, from `rag-datasets/rag-mini-bioasq` (CC BY 2.5). Questions, not paper descriptions, so a loose proxy.
- **local-llm** (253): gemma3:4b writes "describe this paper without its title" for held-out PMIDs, half random, half cited 50+ times in the corpus. Generations sharing any 4-word run with the title were dropped.
- **template** (226): first author + year + two major MeSH terms.
- **claude** (48): landmark-paper descriptions written by Claude, PMIDs checked against corpus titles. Dev use only.

## Models and data

- MedCPT query/article/cross encoders (NCBI, public domain, `ncbi/MedCPT-*`).
- tantivy (MIT) for BM25.
- PubMed baseline from NLM, as distributed for the challenge.

## Licence

MIT.
