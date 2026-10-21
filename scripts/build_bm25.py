"""Build an on-disk BM25 index (tantivy) over title + abstract + MeSH/chemicals/keywords + authors + journal.

One doc per PMID: files are read newest-first and the first copy of a PMID wins.
Usage: python scripts/build_bm25.py [--limit-files N] [--out data/index/bm25]
"""
import argparse, shutil, time
from pathlib import Path

import numpy as np, pyarrow.parquet as pq, tantivy

ROOT = Path(__file__).resolve().parent.parent


def schema():
    sb = tantivy.SchemaBuilder()
    sb.add_integer_field("pmid", stored=True, indexed=True, fast=True)
    for f in ("title", "abstract", "meta"):
        sb.add_text_field(f, stored=False, tokenizer_name="en_stem")
    sb.add_text_field("authors", stored=False, tokenizer_name="default")
    sb.add_text_field("journal", stored=False, tokenizer_name="default")
    sb.add_integer_field("year", stored=True, indexed=True, fast=True)
    return sb.build()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-files", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "data/index/bm25"))
    ap.add_argument("--heap-gb", type=float, default=4)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    idx = tantivy.Index(schema(), path=str(out))
    w = idx.writer(heap_size=int(a.heap_gb * 2**30), num_threads=a.threads)
    files = sorted((ROOT / "data/parquet").glob("pubmed27n*.parquet"), reverse=True)
    if a.limit_files:
        files = files[-a.limit_files:]
    seen = np.zeros(60_000_000, dtype=bool)
    cols = ["pmid", "title", "abstract", "journal", "journal_abbr", "year", "authors", "mesh", "chemicals", "keywords"]
    t0 = time.time(); n = dup = 0
    for i, f in enumerate(files):
        t = pq.read_table(f, columns=cols).to_pylist()
        for r in t:
            p = r["pmid"]
            if seen[p]:
                dup += 1; continue
            seen[p] = True
            w.add_document(tantivy.Document(
                pmid=p, title=r["title"] or "", abstract=r["abstract"] or "",
                meta=" ; ".join((r["mesh"] or []) + (r["chemicals"] or []) + (r["keywords"] or [])),
                authors=" ".join(r["authors"] or []),
                journal=f'{r["journal"] or ""} {r["journal_abbr"] or ""}',
                year=r["year"] or 0))
            n += 1
        if i % 50 == 0:
            print(f"{i}/{len(files)} files  {n} docs  {dup} dups  {time.time()-t0:.0f}s", flush=True)
    w.commit(); w.wait_merging_threads()
    print(f"DONE {n} docs, {dup} superseded dups, {time.time()-t0:.0f}s", flush=True)
