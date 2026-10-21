"""Build the proxy dev set (offline only; nothing here ships in the system).

Sources, one jsonl row each {qid, query, pmid, source, style}:
  public    BioASQ (rag-mini-bioasq test, CC BY 2.5) questions with exactly one gold PMID
  claude    dev/claude_landmark.tsv, landmark descriptions written by Claude (dev only)
  local-llm gemma3:4b via ollama: "describe this paper without its title"
  template  "<first author> <year> <major MeSH terms>"
Held-out targets for local-llm/template: 150 uniform-random PMIDs with abstracts
+ 150 drawn from PMIDs cited >= 50 times inside the corpus (known items skew famous).
"""
import json, random, re, time, urllib.request
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
PQ = str(ROOT / "data/parquet/*.parquet")
OUT = ROOT / "dev/devset.jsonl"
SEED = 20261021
con = duckdb.connect()


def cited_by():
    p = ROOT / "data/cited_by.parquet"
    if not p.exists():
        con.sql(f"copy (select unnest(refs) pmid, count(*)::int n from '{PQ}' group by 1) to '{p}' (format parquet)")
    return str(p)


def articles(pmids):
    rows = con.execute(f"select pmid, title, abstract, year, authors, mesh_major, mesh from '{PQ}' where pmid in (select unnest(?))", [list(pmids)]).fetchall()
    return {r[0]: r for r in rows}


def gram4(s):
    w = re.findall(r"[a-z0-9]+", s.lower())
    return {tuple(w[i:i + 4]) for i in range(len(w) - 3)}


PROMPT = """Below is a biomedical paper. Write ONE search query (15-30 words) that a researcher who remembers this paper, but not its title, would type to find it again. Describe what the paper did or showed in your own words. Do not copy the title or any phrase of 4+ words from it. Output only the query.

Title: {t}
Abstract: {a}"""


def gen(t, a):
    body = {"model": "gemma3:4b", "prompt": PROMPT.format(t=t, a=a[:2000]), "stream": False,
            "options": {"temperature": 0.7, "num_predict": 80, "seed": SEED}}
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        "http://localhost:11434/api/generate", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=120))
    return r["response"].strip().strip('"').splitlines()[0].strip().strip('"')


if __name__ == "__main__":
    rng = random.Random(SEED)
    rows = []
    # public
    bq = con.sql(f"select question, relevant_passage_ids from '{ROOT}/dev/raw/bioasq_test.parquet'").fetchall()
    single = [(q, json.loads(ids)[0]) for q, ids in bq if len(json.loads(ids)) == 1]
    rng.shuffle(single)
    have = articles([p for _, p in single])
    for q, p in single:
        if p in have and sum(r["source"] == "public" for r in rows) < 200:
            rows.append({"query": q, "pmid": p, "source": "public", "style": "bioasq-question"})
    # claude landmark
    lm = [l.split("\t") for l in (ROOT / "dev/claude_landmark.tsv").read_text().splitlines()[1:] if l.strip()]
    have = articles([int(p) for p, _ in lm])
    for p, q in lm:
        if int(p) in have:
            rows.append({"query": q, "pmid": int(p), "source": "claude", "style": "landmark", "title": have[int(p)][1]})
        else:
            print("landmark PMID not in corpus:", p)
    # held-out targets
    cb = cited_by()
    rand = [r[0] for r in con.sql(f"select pmid from (select pmid from '{PQ}' where length(abstract) > 400) using sample 150 rows (reservoir, {SEED % 997})").fetchall()]
    famous = [r[0] for r in con.sql(f"select pmid from (select a.pmid from '{PQ}' a join '{cb}' c using (pmid) where c.n >= 50 and length(a.abstract) > 400) using sample 150 rows (reservoir, {SEED % 991})").fetchall()]
    targets = articles(rand + famous)
    t0 = time.time(); dropped = 0
    for tag, ids in (("random", rand), ("cited50", famous)):
        for p in ids:
            _, t, a, y, au, major, mesh = targets[p]
            q = gen(t, a)
            if gram4(q) & gram4(t):
                q2 = gen(t, a + " ")
                if gram4(q2) & gram4(t):
                    dropped += 1; continue
                q = q2
            rows.append({"query": q, "pmid": p, "source": "local-llm", "style": f"description/{tag}"})
            terms = (major or mesh or [])[:2]
            if au and y and terms:
                rows.append({"query": f"{au[0].split()[0]} {y} {' '.join(terms)}", "pmid": p, "source": "template", "style": f"author-year-mesh/{tag}"})
    print(f"llm gen {time.time()-t0:.0f}s, dropped {dropped} title-copies")
    for i, r in enumerate(rows):
        r["qid"] = f"q{i:04d}"
    OUT.write_text("".join(json.dumps(r) + "\n" for r in rows))
    from collections import Counter
    print(Counter(r["source"] for r in rows))
