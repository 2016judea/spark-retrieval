"""Encode every article with the MedCPT article encoder (fp16, MPS), one shard per parquet file.

Resumable: a shard is written atomically to data/emb/<file>.npy (+ .pmids.npy) and skipped
if it already exists, so a sleep or crash loses at most the shard in progress.
Usage: caffeinate -i -s .venv/bin/python scripts/encode_articles.py [--bs 64] [--newest-first]
"""
import argparse, os, time
from pathlib import Path

import numpy as np, pyarrow.parquet as pq, torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
EMB = ROOT / "data/emb"


def encode_shard(f, tok, model, dev, bs):
    t = pq.read_table(f, columns=["pmid", "title", "abstract"]).to_pydict()
    pmids = np.array(t["pmid"], dtype=np.int64)
    titles = [x or "" for x in t["title"]]
    abstracts = [x or "" for x in t["abstract"]]
    # length-sort to minimise padding; character length is a good enough proxy
    order = np.argsort([len(a) + len(b) for a, b in zip(titles, abstracts)])
    out = np.zeros((len(pmids), 768), dtype=np.float16)
    with torch.no_grad():
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            enc = tok([titles[i] for i in idx], [abstracts[i] for i in idx], truncation="only_second",
                      padding=True, max_length=512, return_tensors="pt").to(dev)
            out[idx] = model(**enc).last_hidden_state[:, 0, :].float().cpu().numpy().astype(np.float16)
    return pmids, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--newest-first", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    EMB.mkdir(parents=True, exist_ok=True)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
    model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder", dtype=torch.float16).to(dev).eval()
    files = sorted((ROOT / "data/parquet").glob("pubmed27n*.parquet"), reverse=a.newest_first)
    todo = [f for f in files if not (EMB / (f.stem + ".npy")).exists()]
    print(f"{len(files) - len(todo)} shards done, {len(todo)} to go", flush=True)
    if a.limit:
        todo = todo[: a.limit]
    t0 = time.time(); ndocs = 0
    for i, f in enumerate(todo):
        ts = time.time()
        pmids, X = encode_shard(f, tok, model, dev, a.bs)
        for name, arr in ((f.stem + ".pmids.npy", pmids), (f.stem + ".npy", X)):  # vectors last = done marker
            tmp = EMB / (name + ".tmp")
            with open(tmp, "wb") as fh:
                np.save(fh, arr)
            os.replace(tmp, EMB / name)
        ndocs += len(pmids)
        rate = ndocs / (time.time() - t0)
        eta = (len(todo) - i - 1) * (time.time() - t0) / (i + 1) / 3600
        print(f"{f.stem} {len(pmids)} docs {time.time()-ts:.0f}s | {i+1}/{len(todo)} shards {rate:.0f} docs/s eta {eta:.1f}h", flush=True)
    print(f"DONE {ndocs} docs in {(time.time()-t0)/3600:.2f}h", flush=True)
