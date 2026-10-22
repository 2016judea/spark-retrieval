"""BM25 search over the tantivy index."""
import re
from pathlib import Path

import tantivy

ROOT = Path(__file__).resolve().parent.parent
STOP = set("""a an and are as at be but by for from has have in into is it its of on or that the their this to was were
which with paper study article describing describes demonstrating showing shows report reported reporting
landmark seminal original first about using used what who how""".split())

FIELDS = ["title", "abstract", "meta", "authors", "journal"]
BOOSTS = {"title": 2.0, "abstract": 1.0, "meta": 1.0, "authors": 1.0, "journal": 0.5}


class BM25:
    def __init__(self, path=ROOT / "data/index/bm25"):
        self.index = tantivy.Index.open(str(path))
        self.index.reload()
        self.searcher = self.index.searcher()

    @staticmethod
    def clean(q):
        toks = re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*", q)
        out = []
        for t in toks:
            for s in t.split("-"):
                if s and s.lower() not in STOP:
                    out.append(s)
        return " ".join(out)

    def search(self, q, k=100, boosts=BOOSTS):
        qq = self.clean(q)
        if not qq:
            return []
        query = self.index.parse_query(qq, FIELDS, field_boosts=boosts)
        res = self.searcher.search(query, k)
        return [(self.searcher.doc(addr)["pmid"][0], s) for s, addr in res.hits]


def rrf(lists, k=60, weights=None):
    """Reciprocal-rank fusion of ranked PMID lists."""
    from collections import defaultdict
    sc = defaultdict(float)
    for i, lst in enumerate(lists):
        w = weights[i] if weights else 1.0
        for r, p in enumerate(lst, 1):
            sc[p] += w / (k + r)
    return [p for p, _ in sorted(sc.items(), key=lambda x: -x[1])]


def search_rewritten(b, q, rewrites, k=1000, k_rw=300, orig_w=1.0, mode="rrf"):
    """BM25 on the original question plus each open-model rewrite.
    mode rrf:    one BM25 list per query, fused by RRF.
    mode expand: one BM25 query = original + all rewrites (keeps the original's rare terms).
    mode both:   RRF of the original list and the expanded list."""
    if mode in ("expand", "both"):
        ex = [p for p, _ in b.search(q + " " + " ".join(rewrites), k)] if rewrites else [p for p, _ in b.search(q, k)]
        if mode == "expand":
            return ex
        return rrf([[p for p, _ in b.search(q, k)], ex], weights=[orig_w, 1.0])
    lists = [[p for p, _ in b.search(q, k)]] + [[p for p, _ in b.search(r, k_rw)] for r in rewrites]
    return rrf(lists, weights=[orig_w] + [1.0] * len(rewrites))
