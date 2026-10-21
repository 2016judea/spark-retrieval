"""BM25 + MedCPT dense fusion (RRF), then MedCPT cross-encoder rerank. MRR@10 per dev set.

Usage: python scripts/eval_fusion.py [--ann] [--k 1000] [--rerank 100]
  --ann  use the IVF-PQ index (full corpus). Without it, dense search is exact over the
         shards encoded so far (partial corpus; for testing the pipeline only).
Score variants reported:
  bm25 | dense | rrf | rrf>ce | rrf>ce+meta (passage prefixed with authors/journal/year)
  | rrf>ce+meta+prior (year-match and citation prior; weights fit on even qids, reported on odd)
"""
import argparse, json, re, sys, time
from collections import defaultdict
from pathlib import Path

import duckdb, numpy as np, torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from spark.bm25 import BM25
from spark.dense import INDEX, QueryEncoder, Store

RRF_K = 60


def mrr10(ranks):
    return sum(1 / r for r in ranks if r and r <= 10) / len(ranks)


def rank_of(lst, gold):
    try:
        return lst.index(gold) + 1
    except ValueError:
        return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ann", action="store_true")
    ap.add_argument("--k", type=int, default=1000); ap.add_argument("--rerank", type=int, default=100)
    ap.add_argument("--nprobe", type=int, default=64); ap.add_argument("--tag", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--index", default=str(INDEX))
    a = ap.parse_args()
    dev = [json.loads(l) for l in (ROOT / "dev/devset.jsonl").read_text().splitlines()]
    if a.limit:
        dev = dev[:: max(1, len(dev) // a.limit)][: a.limit]
    T = {}

    # 1. BM25
    t0 = time.time(); b = BM25()
    bm = [[p for p, _ in b.search(r["query"], a.k)] for r in dev]
    T["bm25_ms"] = (time.time() - t0) / len(dev) * 1000

    # 2. dense
    st = Store()
    print(f"dense store: {len(st)} vectors from {len(st.shards)} shards", flush=True)
    t0 = time.time(); Q = QueryEncoder()(([r["query"] for r in dev])); T["qenc_ms"] = (time.time() - t0) / len(dev) * 1000
    t0 = time.time()
    if a.ann:  # faiss in a subprocess: faiss + torch in one process aborts on macOS (libomp)
        import subprocess
        qf, of = ROOT / "runs/dev_q.npy", ROOT / "runs/dev_ann.npy"
        qf.parent.mkdir(exist_ok=True); np.save(qf, Q)
        r = subprocess.run([sys.executable, str(ROOT / "scripts/ann_search.py"), str(qf), str(of),
                            "--k", str(a.k), "--nprobe", str(a.nprobe), "--index", a.index], capture_output=True, text=True)
        print(r.stdout.strip(), r.stderr.strip()[-500:] if r.returncode else "", flush=True)
        r.check_returncode()
        I = np.load(of)
    else:  # exact, shard by shard, keep running top-k
        best_s = np.full((len(dev), 0), -np.inf, np.float32); best_i = np.zeros((len(dev), 0), np.int64)
        for s, v in enumerate(st.vecs):
            S = Q @ np.asarray(v, dtype=np.float32).T
            k = min(a.k, S.shape[1]); top = np.argpartition(-S, k - 1, axis=1)[:, :k]
            best_s = np.concatenate([best_s, np.take_along_axis(S, top, 1)], 1)
            best_i = np.concatenate([best_i, top + st.offsets[s]], 1)
            keep = np.argsort(-best_s, 1)[:, : a.k]
            best_s = np.take_along_axis(best_s, keep, 1); best_i = np.take_along_axis(best_i, keep, 1)
        I = best_i
    T["ann_ms"] = (time.time() - t0) / len(dev) * 1000

    # 3. exact dense rescoring of the candidate union, then RRF
    t0 = time.time(); fused, dense_ranked = [], []
    for qi, r in enumerate(dev):
        drows = [x for x in I[qi] if x >= 0]
        dp = st.pmids[drows].tolist()
        cand = list(dict.fromkeys(bm[qi] + dp))
        rows = st.rows_for(cand)
        sc = np.full(len(cand), -np.inf, np.float32)
        ok = rows >= 0
        if ok.any():
            sc[ok] = st.get(rows[ok]) @ Q[qi]
        dorder = [cand[i] for i in np.argsort(-sc) if np.isfinite(sc[i])]
        dense_ranked.append(dorder)
        rr = defaultdict(float)
        for lst in (bm[qi], dorder[: a.k]):
            for rank, p in enumerate(lst, 1):
                rr[p] += 1 / (RRF_K + rank)
        fused.append([p for p, _ in sorted(rr.items(), key=lambda x: -x[1])])
    T["fuse_ms"] = (time.time() - t0) / len(dev) * 1000

    # 4. cross-encoder over the top of the fused list
    need = sorted({p for f in fused for p in f[: a.rerank]})
    meta = {r[0]: r[1:] for r in duckdb.execute(
        f"select pmid, title, abstract, authors, journal_abbr, year from '{ROOT}/data/parquet/*.parquet' where pmid in (select unnest(?))", [need]).fetchall()}
    cited = dict(duckdb.execute(f"select pmid, n from '{ROOT}/data/cited_by.parquet' where pmid in (select unnest(?))", [need]).fetchall())
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("ncbi/MedCPT-Cross-Encoder")
    ce = AutoModelForSequenceClassification.from_pretrained("ncbi/MedCPT-Cross-Encoder", dtype=torch.float16).to("mps").eval()

    def plain(p):
        t, ab, *_ = meta.get(p, ("", "", [], "", None)); return f"{t} {ab}"

    def withmeta(p):
        t, ab, au, j, y = meta.get(p, ("", "", [], "", None))
        au = au or []
        who = ", ".join(au[:3]) + (" et al." if len(au) > 3 else "")
        return f"{who}. {j} {y or ''}. {t} {ab}"

    def ce_scores(q, cands, fmt):
        out = []
        with torch.no_grad():
            for s in range(0, len(cands), 50):
                enc = tok([[q, fmt(p)] for p in cands[s:s + 50]], truncation=True, padding=True, max_length=512, return_tensors="pt").to("mps")
                out += ce(**enc).logits.squeeze(-1).float().cpu().tolist()
        return np.array(out)

    t0 = time.time(); CE = {"plain": [], "meta": []}
    for qi, r in enumerate(dev):
        c = fused[qi][: a.rerank]
        CE["plain"].append(ce_scores(r["query"], c, plain))
        CE["meta"].append(ce_scores(r["query"], c, withmeta))
    T["ce_ms_per_variant"] = (time.time() - t0) / len(dev) * 1000 / 2

    # 5. prior: year mentioned in query matches, log citations. Fit on even qids, report on odd.
    def feats(qi):
        c = fused[qi][: a.rerank]
        yrs = {int(y) for y in re.findall(r"\b(19[5-9]\d|20[0-2]\d)\b", dev[qi]["query"])}
        ym = np.array([1.0 if yrs and meta.get(p, (0, 0, 0, 0, None))[4] in yrs else 0.0 for p in c])
        lc = np.log1p(np.array([cited.get(p, 0) for p in c], dtype=np.float32))
        return c, ym, lc

    F = [feats(qi) for qi in range(len(dev))]

    def ranks_prior(wy, wc, qids):
        out = []
        for qi in qids:
            c, ym, lc = F[qi]
            s = CE["meta"][qi] + wy * ym + wc * lc
            out.append(rank_of([c[i] for i in np.argsort(-s)], dev[qi]["pmid"]))
        return out

    even = [i for i in range(len(dev)) if i % 2 == 0]
    grid = [(wy, wc) for wy in (0, 1, 2, 4, 8) for wc in (0, 0.1, 0.25, 0.5, 1.0)]
    wy, wc = max(grid, key=lambda w: mrr10(ranks_prior(*w, even)))

    # 6. report
    variants = {
        "bm25": [rank_of(bm[i], d["pmid"]) for i, d in enumerate(dev)],
        "dense": [rank_of(dense_ranked[i], d["pmid"]) for i, d in enumerate(dev)],
        "rrf": [rank_of(fused[i], d["pmid"]) for i, d in enumerate(dev)],
    }
    for v in ("plain", "meta"):
        variants[f"rrf>ce{'' if v == 'plain' else '+meta'}"] = [
            rank_of([fused[i][: a.rerank][j] for j in np.argsort(-CE[v][i])], d["pmid"]) for i, d in enumerate(dev)]
    variants["rrf>ce+meta+prior"] = ranks_prior(wy, wc, range(len(dev)))
    groups = defaultdict(list)
    for i, d in enumerate(dev):
        groups["ALL"].append(i); groups[d["source"]].append(i)
    odd = [i for i in range(len(dev)) if i % 2 == 1]
    names = ["ALL", "public", "local-llm", "template", "claude"]
    lines = [f"{'variant':20s} " + " ".join(f"{g:>9s}" for g in names) + f" {'ALL-odd':>8s}"]
    res = {}
    for v, rk in variants.items():
        row = {g: mrr10([rk[i] for i in groups[g]]) for g in names}
        row["ALL-odd"] = mrr10([rk[i] for i in odd]); res[v] = row
        lines.append(f"{v:20s} " + " ".join(f"{row[g]:9.3f}" for g in names) + f" {row['ALL-odd']:8.3f}")
    rec = lambda lists, k: sum(1 for i, d in enumerate(dev) if d["pmid"] in lists[i][:k]) / len(dev)
    lines.append(f"recall@{a.rerank} (rerank ceiling): bm25 {rec(bm, a.rerank):.3f}  dense {rec(dense_ranked, a.rerank):.3f}  rrf {rec(fused, a.rerank):.3f}")
    lines.append(f"prior weights fit on even qids: year {wy}, log-cites {wc}")
    lines.append("ms/query: " + ", ".join(f"{k} {v:.0f}" for k, v in T.items()))
    lines.append(f"corpus: {len(st)} dense vectors ({'IVF-PQ nprobe %d' % a.nprobe if a.ann else 'exact, partial'})")
    print("\n".join(lines))
    (ROOT / "runs").mkdir(exist_ok=True)
    (ROOT / f"runs/fusion{a.tag}.json").write_text(json.dumps({"mrr10": res, "prior": [wy, wc], "ms": T, "n_dense": len(st), "ann": a.ann, "report": lines}, indent=1))
