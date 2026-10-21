"""Dense side: shard store of fp16 MedCPT article vectors, IVF-PQ ANN, MedCPT query encoder."""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EMB = ROOT / "data/emb"
INDEX = ROOT / "data/index/dense_ivfpq.faiss"


class Store:
    """Global row id = position in the concatenation of finished shards, oldest file first."""

    def __init__(self, emb=EMB):
        self.shards = sorted(p for p in emb.glob("pubmed27n????.npy"))
        self.vecs = [np.load(p, mmap_mode="r") for p in self.shards]
        pm = [np.load(str(p)[:-4] + ".pmids.npy") for p in self.shards]
        self.sizes = np.array([len(x) for x in pm])
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)])
        self.pmids = np.concatenate(pm) if pm else np.zeros(0, np.int64)
        self.row_of = np.full(int(self.pmids.max()) + 1 if len(self.pmids) else 1, -1, dtype=np.int64)
        # newest file wins for a PMID that appears twice
        self.row_of[self.pmids] = np.arange(len(self.pmids))

    def __len__(self):
        return len(self.pmids)

    def get(self, rows):
        rows = np.asarray(rows)
        out = np.zeros((len(rows), 768), dtype=np.float32)
        sh = np.searchsorted(self.offsets, rows, side="right") - 1
        for s in np.unique(sh):
            m = sh == s
            out[m] = self.vecs[s][rows[m] - self.offsets[s]]
        return out

    def rows_for(self, pmids):
        pmids = np.asarray(pmids)
        ok = pmids < len(self.row_of)
        r = np.full(len(pmids), -1, dtype=np.int64)
        r[ok] = self.row_of[pmids[ok]]
        return r


class QueryEncoder:
    def __init__(self, device=None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.dev = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
        self.m = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to(self.dev).eval()

    def __call__(self, queries, bs=64):
        out = []
        with self.torch.no_grad():
            for s in range(0, len(queries), bs):
                enc = self.tok(queries[s:s + bs], truncation=True, padding=True, max_length=64, return_tensors="pt").to(self.dev)
                out.append(self.m(**enc).last_hidden_state[:, 0, :].float().cpu().numpy())
        return np.concatenate(out)


def load_ann(path=INDEX, nprobe=64):
    import faiss
    idx = faiss.read_index(str(path))
    faiss.extract_index_ivf(idx).nprobe = nprobe
    return idx
