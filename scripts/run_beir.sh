#!/usr/bin/env bash
# Does the outer-pooling gain hold beyond one dataset? Small BEIR corpora only:
# scoring is exhaustive and token-level, so the corpus has to fit in HBM.
set -uo pipefail
cd "$(dirname "$0")/.."
for d in arguana scidocs fiqa; do
  echo "=== $d ==="
  .venv/bin/python -u scripts/late_interaction.py --dataset "$d" 2>&1 \
    | grep -v "Loading weights\|^\[transformers\]\|UNEXPECTED\|ignored when\|^Key \|^-----\|^Notes:"
done
echo BEIRDONE
