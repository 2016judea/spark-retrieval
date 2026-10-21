"""Time MedCPT article encoding, query encoding and cross-encoder reranking on this machine.

Usage: python scripts/bench_dense.py [--n 2048]
Samples abstracts from the most recent parsed parquet files (modern abstract lengths).
"""
import argparse, json, platform, statistics, time
from pathlib import Path

import duckdb, numpy as np, torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent


def sync(dev):
    if dev == "mps":
        torch.mps.synchronize()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=2048)
    a = ap.parse_args()
    files = sorted((ROOT / "data/parquet").glob("*.parquet"))[-5:]
    rows = duckdb.sql(f"select title, abstract from read_parquet({[str(f) for f in files]}) where length(abstract)>0 using sample {a.n} rows (reservoir, 1)").fetchall()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    out = {"device": dev, "machine": platform.machine(), "n_docs": len(rows), "src_files": [f.name for f in files]}

    tok = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
    lens = [len(x) for x in tok([t for t, _ in rows], [b for _, b in rows], truncation="only_second", max_length=512)["input_ids"]]
    out["article_tokens_median"] = statistics.median(lens); out["article_tokens_p90"] = sorted(lens)[int(.9 * len(lens))]

    for dtype in (torch.float16, torch.float32):
        m = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder", dtype=dtype).to(dev).eval()
        for bs in (32, 64):
            # length-sorted batching, as the real indexer will do
            order = np.argsort(lens)
            with torch.no_grad():
                b = [rows[i] for i in order[:bs]]
                enc = tok([t for t, _ in b], [x for _, x in b], truncation="only_second", padding=True, max_length=512, return_tensors="pt").to(dev)
                m(**enc); sync(dev)  # warmup
                t0 = time.time()
                for s in range(0, len(rows), bs):
                    b = [rows[i] for i in order[s:s + bs]]
                    enc = tok([t for t, _ in b], [x for _, x in b], truncation="only_second", padding=True, max_length=512, return_tensors="pt").to(dev)
                    e = m(**enc).last_hidden_state[:, 0, :]
                sync(dev)
            dt = time.time() - t0
            out[f"article_docs_per_s_{str(dtype)[6:]}_bs{bs}"] = round(len(rows) / dt, 1)
            print(dtype, bs, round(len(rows) / dt, 1), "docs/s", flush=True)
        del m

    qtok = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
    qm = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to(dev).eval()
    qs = [t for t, _ in rows[:200]]
    lat = []
    with torch.no_grad():
        for q in qs:
            t0 = time.time(); enc = qtok([q], truncation=True, max_length=64, return_tensors="pt").to(dev)
            qm(**enc).last_hidden_state[:, 0, :].cpu(); lat.append(time.time() - t0)
    out["query_encode_ms_median"] = round(statistics.median(lat[10:]) * 1000, 1)

    ctok = AutoTokenizer.from_pretrained("ncbi/MedCPT-Cross-Encoder")
    for dtype in (torch.float16, torch.float32):
        cm = AutoModelForSequenceClassification.from_pretrained("ncbi/MedCPT-Cross-Encoder", dtype=dtype).to(dev).eval()
        for k in (100, 200):
            pairs = [[qs[0], t + " " + x] for t, x in (rows * 4)[:k]]
            with torch.no_grad():
                times = []
                for _ in range(4):
                    t0 = time.time()
                    for s in range(0, k, 50):
                        enc = ctok(pairs[s:s + 50], truncation=True, padding=True, max_length=512, return_tensors="pt").to(dev)
                        cm(**enc).logits.squeeze(-1).float().cpu()
                    sync(dev); times.append(time.time() - t0)
            out[f"rerank_top{k}_ms_{str(dtype)[6:]}"] = round(statistics.median(times[1:]) * 1000)
            print("rerank", dtype, k, out[f"rerank_top{k}_ms_{str(dtype)[6:]}"], "ms", flush=True)
        del cm
    (ROOT / "runs").mkdir(exist_ok=True)
    (ROOT / "runs/bench_dense.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
