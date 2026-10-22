"""Cache open-model rewrites for every dev + official query.
Usage: python scripts/gen_rewrites.py MODEL PROMPT   -> runs/rewrites/<model>_<prompt>.jsonl (resumable)
"""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from spark.rewrite import rewrite

model, prompt = sys.argv[1], sys.argv[2]
out = ROOT / f"runs/rewrites/{model.replace(':', '-')}_{prompt}.jsonl"
done = {json.loads(l)["qid"] for l in out.read_text().splitlines()} if out.exists() else set()
rows = [json.loads(l) for f in ("dev/devset.jsonl", "dev/official.jsonl") for l in (ROOT / f).read_text().splitlines()]
with out.open("a") as fh:
    for r in rows:
        if r["qid"] in done:
            continue
        t = time.time(); qs = rewrite(r["query"], model, prompt); ms = (time.time() - t) * 1000
        fh.write(json.dumps({"qid": r["qid"], "rewrites": qs, "ms": round(ms)}) + "\n"); fh.flush()
print(out, "done")
