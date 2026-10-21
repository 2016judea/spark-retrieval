# Plan: dense retrieval + rerank, with measured costs on this Mac

Machine: Apple M5 Pro, 24 GB RAM, MPS. All numbers from `runs/` (2026-10-21); commands beside each.

## Where we are (measured)

| stage | MRR@10 (727 dev queries) | latency/query | command |
|---|---|---|---|
| BM25 (tantivy, title x2 + abstract + MeSH/chem/keywords + authors + journal) | **0.414** | 112 ms median, 269 ms p95 (k=1000) | `scripts/eval.py` |
| BM25 top-100 -> MedCPT cross-encoder (fp16) | **0.500** | +425 ms | `scripts/eval_rerank.py --k 100` |

Per subset, BM25 -> rerank: public BioASQ 0.459 -> 0.542, local-LLM descriptions 0.628 -> 0.853,
landmark (Claude-written, title-leaning) 0.616 -> 0.609, author+year+MeSH templates 0.090 -> 0.047.

The ceiling for any reranker on BM25 candidates is BM25 recall: R@100 0.682, R@1000 0.816
(public 0.745 / 0.840). Dense retrieval exists to lift that ceiling.

## Stage 3: dense retrieval (MedCPT article/query encoders, public domain, NCBI)

Measured (`scripts/bench_dense.py`, 1,862 abstracts from the 2026 files, median 343 tokens, p90 484):

- Article encoder fp16 on MPS: **349 docs/s** (bs 64); fp32 98 docs/s. fp16 vs fp32 cosine = 1.0000 on 64 docs, so fp16.
- Query encoder: 6.4 ms/query.

Cost to embed the corpus (39,167,523 PMIDs; 74.1% have an abstract):
- 29.0M with abstracts / 349 per s = **~23 h** of MPS time.
- 10.1M title-only: not measured; at most 8 h at the abstract rate, likely far less (titles are ~20 tokens).
- So **one overnight-plus run, 24-31 h**, under `caffeinate -i -s`, resumable per parquet file.

Storage: 768-d fp16 = 1,536 B/doc -> **60 GB on disk** (264 GB free now). Does not fit in 24 GB RAM, so two options:
1. **ANN**: faiss IVF65536,PQ96 = 96 B/doc -> **3.8 GB in RAM**. Training + adding ~1-2 h CPU (not yet measured). Searches all 39M.
2. **Exact on candidates**: memmap the fp16 matrix and score only BM25's top-1000 (1.5 MB random read per query). No new recall, so it is the cheap fallback.
Do (1), fuse with BM25 by RRF, measure R@100 and MRR@10; keep (2) only if (1) adds nothing.

## Stage 4: rerank

- MedCPT cross-encoder fp16: 440 ms per 100 candidates, 892 ms per 200 (linear). Top-100 is the default; widen to 200 only if fused R@200 - R@100 pays for +450 ms.
- Fix measured above: the template queries *drop* under rerank because the passage carries no authors/year/journal. Put "Author A, Author B, Journal, Year" in front of the title in the passage, and parse a 4-digit year out of the query as a soft filter in BM25.
- Landmark prior: in-corpus cited-by counts already built (`data/cited_by.parquet`, 418M references, 30% of records carry a reference list). Add log(1+cited_by) as a feature in the final score; it costs nothing at query time.

## Stage 5: Docker

Image = code + MedCPT (3 x 438 MB) + tantivy index (16 GB) + PQ index (~4 GB) + a slim article store for rerank text.
Expected memory at query time: < 8 GB. Measure it in the container.

## Disk and time used so far

- Download + md5 verify: 1,306/1,306 files OK, 37 min, 54 GB.
- Parse to parquet: ~9 min total on 10 workers, 21 GB, 39,167,524 rows (1 superseded duplicate).
- BM25 build: 6 min (370 s), 16 GB.
- Total on disk: 92 GB under `data/` (gitignored).
