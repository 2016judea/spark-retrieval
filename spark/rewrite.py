"""Open-weight query rewriting: a small local model (ollama) turns a question into 2-4
keyword-style BM25 queries. Ships in the image; no network beyond localhost.

    from spark.rewrite import rewrite
    rewrite("Which publication first introduced AlphaFold?", model="gemma3:4b")
    -> ["AlphaFold protein structure prediction", "deep learning protein folding CASP13", ...]
"""
import json
import re
import urllib.request

PROMPTS = {
    # v1: four labelled facets
    "facets": """You help find one biomedical paper in PubMed. Rewrite the user's description into 2-4 short keyword search queries (3-10 words each, no question words, no filler):
1. the paper's likely TITLE phrasing
2. the key entities: genes, proteins, drugs, organisms, diseases, methods, trial names
3. MeSH-style subject terms and synonyms
4. author surname, journal and year ONLY if they are stated or you are certain of them; otherwise skip
Return JSON: {{"queries": ["...", "..."]}}

Description: {q}""",
    # v2: guess the title outright (the model often knows famous papers)
    "title": """A researcher is looking for one specific biomedical paper in PubMed and describes it below.
Write 3 different guesses at the paper's actual TITLE, as it would appear in PubMed (technical wording, specific entities, no question words). If you know the exact paper, put its real title first.
Return JSON: {{"queries": ["title guess 1", "title guess 2", "title guess 3"]}}

Description: {q}""",
    # v3: recall the citation; names, trial acronyms and title words are what BM25 can match
    "cite": """A researcher is looking for one specific biomedical paper in PubMed and describes it below.
Recall the paper. Give up to 3 search queries for PubMed:
1. your best guess at its exact TITLE
2. first author surname, journal and year (e.g. "Smith Nature 1998"), only if you know the paper
3. the specific names a title or abstract would contain: trial acronym, gene, drug, organism, method (keywords only)
Return JSON: {{"queries": ["...", "...", "..."]}}

Description: {q}""",
}


def rewrite(q, model="gemma3:4b", prompt="facets", host="http://localhost:11434", timeout=120):
    body = {"model": model, "prompt": PROMPTS[prompt].format(q=q), "stream": False, "format": "json",
            "options": {"temperature": 0, "num_predict": 160, "seed": 0}}
    if model.startswith("qwen3"):
        body["think"] = False
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        f"{host}/api/generate", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=timeout))
    txt = r["response"]
    try:
        obj = json.loads(txt)
        qs = obj.get("queries", []) if isinstance(obj, dict) else obj
    except json.JSONDecodeError:
        qs = re.findall(r'"([^"]{6,200})"', txt)
    out = []
    for s in qs:
        if isinstance(s, str) and s.strip() and s.strip().lower() != q.strip().lower():
            out.append(s.strip())
    return out[:4]
