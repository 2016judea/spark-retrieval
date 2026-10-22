"""BM25-only screen of rewrite configs: recall@100 and MRR@10 per dev set, original vs +rewrites.
Usage: python scripts/eval_rewrite_bm25.py runs/rewrites/A.jsonl [B.jsonl ...]  (several files = union of their rewrites)
       [--orig-w 1.0] [--tag name]
"""
import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from spark.bm25 import BM25, search_rewritten

ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); ap.add_argument("--orig-w", type=float, default=1.0); ap.add_argument("--mode", default="rrf")
a = ap.parse_args()
rows = [json.loads(l) for f in ("dev/devset.jsonl", "dev/official.jsonl") for l in (ROOT / f).read_text().splitlines()]
rw = defaultdict(list); gen_ms = defaultdict(float)
for f in a.files:
    for l in Path(f).read_text().splitlines():
        d = json.loads(l); rw[d["qid"]] += d["rewrites"]; gen_ms[d["qid"]] += d["ms"]
rows = [r for r in rows if r["qid"] in rw]
b = BM25()
cache = ROOT / "runs/bm25_orig_k1000.json"
base = json.loads(cache.read_text()) if cache.exists() else {}
t0 = time.time(); ranks = {"bm25": [], "bm25+rw": []}
for r in rows:
    if r["qid"] not in base:
        base[r["qid"]] = [p for p, _ in b.search(r["query"], 1000)]
    o = base[r["qid"]]
    f = search_rewritten(b, r["query"], rw[r["qid"]], orig_w=a.orig_w, mode=a.mode)
    for name, lst in (("bm25", o), ("bm25+rw", f)):
        ranks[name].append(lst.index(r["pmid"]) + 1 if r["pmid"] in lst else None)
cache.write_text(json.dumps(base))
ms = (time.time() - t0) / len(rows) * 1000
names = ["ALL", "public", "local-llm", "template", "claude", "official"]
G = defaultdict(list)
for i, r in enumerate(rows):
    G[r["source"]].append(i)
    if r["source"] != "official": G["ALL"].append(i)
print(f"files {[Path(f).name for f in a.files]} orig_w {a.orig_w} mode {a.mode}  n={len(rows)}  gen {sum(gen_ms.values())/len(gen_ms):.0f} ms/q  search {ms:.0f} ms/q")
for name, rk in ranks.items():
    rec = {g: sum(1 for i in G[g] if rk[i] and rk[i] <= 100) / len(G[g]) for g in names if G[g]}
    mrr = {g: sum(1 / rk[i] for i in G[g] if rk[i] and rk[i] <= 10) / len(G[g]) for g in names if G[g]}
    print(f"  {name:8s} R@100 " + " ".join(f"{g}={rec[g]:.3f}" for g in rec))
    print(f"  {name:8s} MRR10 " + " ".join(f"{g}={mrr[g]:.3f}" for g in mrr))
print("  official ranks:", {rows[i]["qid"]: (ranks['bm25'][i], ranks['bm25+rw'][i]) for i in G["official"]})
