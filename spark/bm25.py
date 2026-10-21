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
