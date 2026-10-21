"""faiss-only ANN search: query vectors in, row ids out.

Runs as its own process because faiss and torch each bundle libomp and abort when
both load in one process on macOS (OMP Error #15, seen 2026-10-21).
Usage: python scripts/ann_search.py Q.npy OUT.npy [--k 1000] [--nprobe 64] [--index path]
"""
import argparse, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from spark.dense import INDEX, load_ann

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("q"); ap.add_argument("out")
    ap.add_argument("--k", type=int, default=1000); ap.add_argument("--nprobe", type=int, default=64)
    ap.add_argument("--index", default=str(INDEX))
    a = ap.parse_args()
    t0 = time.time(); idx = load_ann(Path(a.index), a.nprobe); tl = time.time() - t0
    Q = np.load(a.q).astype(np.float32)
    t0 = time.time(); _, I = idx.search(Q, a.k); ts = time.time() - t0
    np.save(a.out, I)
    print(f"ann: load {tl:.0f}s, search {ts / len(Q) * 1000:.1f} ms/query over {idx.ntotal} vectors", flush=True)
