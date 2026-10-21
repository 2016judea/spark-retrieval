#!/usr/bin/env bash
# Download the frozen SPARK PubMed baseline and md5-verify every file.
# Idempotent: a file whose md5 already matches is skipped. 6 parallel fetches.
# Usage: caffeinate -i -s scripts/download_baseline.sh
set -euo pipefail
BASE=https://ftp.ncbi.nlm.nih.gov/pubmed/.pubmed-spark
DIR="$(cd "$(dirname "$0")/.." && pwd)/data/baseline"
mkdir -p "$DIR"
cd "$DIR"

curl -s "$BASE/" | grep -o 'pubmed27n[0-9]*\.xml\.gz"' | tr -d '"' | sort -u > files.txt
echo "listed $(wc -l < files.txt) files"

fetch() {
  f="$1"
  for attempt in 1 2 3 4 5; do
    curl -s --retry 3 -o "$f.md5" "$BASE/$f.md5"
    want=$(sed -E 's/.*= *([0-9a-f]{32}).*/\1/' "$f.md5")
    if [ -f "$f" ] && [ "$(md5 -q "$f")" = "$want" ]; then echo "OK $f"; return 0; fi
    curl -s --retry 3 -o "$f.part" "$BASE/$f" && mv "$f.part" "$f"
    if [ "$(md5 -q "$f")" = "$want" ]; then echo "OK $f"; return 0; fi
    echo "BADMD5 $f attempt $attempt" >&2
  done
  echo "FAIL $f"
}
export -f fetch; export BASE
xargs -P 6 -I{} bash -c 'fetch {}' < files.txt > verify.log
echo "verified: $(grep -c '^OK' verify.log)  failed: $(grep -c '^FAIL' verify.log || true)"
