#!/usr/bin/env bash
# Wait for the encode to finish, then build the ANN index, run the full fusion eval, commit results.
cd "$(dirname "$0")/.."
until grep -q ALL_SHARDS_DONE logs/encode.log; do sleep 60; done
.venv/bin/python scripts/build_dense_index.py > logs/dense_index.log 2>&1 || { echo INDEX_FAILED; exit 1; }
.venv/bin/python scripts/eval_fusion.py --ann 2>&1 | grep -vE 'Warning|Loading weights|Redirects' > logs/fusion.log || { echo EVAL_FAILED; exit 1; }
git add runs/fusion.json && git commit -qm "Full-corpus fusion eval: MRR@10 per dev set (runs/fusion.json)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- runs/fusion.json && git push -q
echo CHAIN_DONE
