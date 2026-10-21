"""Build the IVF-PQ ANN index over all encoded shards (inner product, MedCPT's similarity).

Usage: python scripts/build_dense_index.py [--nlist 16384] [--m 96] [--train 1000000]
IDs stored in the index are global Store rows (see spark/dense.py).
"""
import argparse, sys, time
from pathlib import Path

import faiss, numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from spark.dense import INDEX, Store

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nlist", type=int, default=16384); ap.add_argument("--m", type=int, default=96)
    ap.add_argument("--train", type=int, default=1_000_000); ap.add_argument("--out", default=str(INDEX))
    a = ap.parse_args()
    faiss.omp_set_num_threads(12)
    st = Store(); n = len(st)
    print(f"{n} vectors in {len(st.shards)} shards", flush=True)
    rng = np.random.default_rng(0)
    t0 = time.time()
    sample = np.sort(rng.choice(n, size=min(a.train, n), replace=False))
    xt = st.get(sample)
    idx = faiss.index_factory(768, f"IVF{a.nlist},PQ{a.m}", faiss.METRIC_INNER_PRODUCT)
    idx.train(xt); del xt
    print(f"trained in {time.time()-t0:.0f}s", flush=True)
    t1 = time.time()
    for s, v in enumerate(st.vecs):
        idx.add_with_ids(np.asarray(v, dtype=np.float32), np.arange(st.offsets[s], st.offsets[s + 1], dtype=np.int64))
        if s % 100 == 0:
            print(f"added {s}/{len(st.vecs)} shards {time.time()-t1:.0f}s", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(idx, a.out)
    print(f"DONE {idx.ntotal} vectors, add {time.time()-t1:.0f}s, total {time.time()-t0:.0f}s, "
          f"{Path(a.out).stat().st_size/2**30:.1f} GB", flush=True)
