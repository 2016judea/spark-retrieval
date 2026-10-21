#!/usr/bin/env bash
# Run the article encode to completion, restarting after a crash. Resumes by shard.
# Usage: nohup caffeinate -i -s scripts/encode_all.sh > logs/encode.log 2>&1 &
cd "$(dirname "$0")/.."
for attempt in $(seq 1 20); do
  .venv/bin/python scripts/encode_articles.py 2>&1 | grep --line-buffered -vE 'Warning|Loading weights|Redirects'
  [ "$(ls data/emb/*.pmids.npy 2>/dev/null | wc -l)" -ge "$(ls data/parquet/pubmed27n*.parquet | wc -l)" ] && \
    [ "$(ls data/emb/pubmed27n????.npy | wc -l)" -ge "$(ls data/parquet/pubmed27n*.parquet | wc -l)" ] && { echo ALL_SHARDS_DONE; exit 0; }
  echo "RESTART attempt $attempt"; sleep 30
done
