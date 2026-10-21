"""MRR@10 / Recall@k of a retriever on dev/devset.jsonl, per source.

Usage: python scripts/eval.py [--k 1000] [--run-out runs/bm25.jsonl]
"""
import argparse, json, statistics, sys, time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from spark.bm25 import BM25

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--run-out", default=str(ROOT / "runs/bm25.jsonl"))
    a = ap.parse_args()
    dev = [json.loads(l) for l in (ROOT / "dev/devset.jsonl").read_text().splitlines()]
    b = BM25()
    by = defaultdict(list); lat = []; run = []
    for r in dev:
        t0 = time.time(); hits = b.search(r["query"], a.k); lat.append(time.time() - t0)
        ids = [p for p, _ in hits]
        rank = ids.index(r["pmid"]) + 1 if r["pmid"] in ids else None
        run.append({"qid": r["qid"], "rank": rank, "top": ids[:100]})
        for key in ("ALL", r["source"], f'{r["source"]}:{r["style"]}'):
            by[key].append(rank)
    Path(a.run_out).parent.mkdir(exist_ok=True)
    Path(a.run_out).write_text("".join(json.dumps(x) + "\n" for x in run))
    print(f"{'subset':38s} {'n':>4s} {'MRR@10':>7s} {'R@1':>6s} {'R@10':>6s} {'R@100':>6s} {'R@1000':>6s}")
    for key in sorted(by, key=lambda k: (k != "ALL", k)):
        rs = by[key]; n = len(rs)
        mrr = sum(1 / x for x in rs if x and x <= 10) / n
        rec = lambda k: sum(1 for x in rs if x and x <= k) / n
        print(f"{key:38s} {n:4d} {mrr:7.3f} {rec(1):6.3f} {rec(10):6.3f} {rec(100):6.3f} {rec(1000):6.3f}")
    lat.sort()
    print(f"latency per query: median {statistics.median(lat)*1000:.0f} ms, p95 {lat[int(len(lat)*.95)]*1000:.0f} ms (k={a.k})")
