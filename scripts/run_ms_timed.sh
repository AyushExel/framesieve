#!/usr/bin/env bash
# Re-run the three MomentSeeker conditions with the timing decomposition, so the
# cost table separates model time from frame-fetch time (an implementation
# artifact with a known 14x fix) rather than conflating them.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
P=.venv/bin/python
$P -u scripts/eval_momentseeker.py --vlm-budgets 0                    --out runs/ms_t0.json
$P -u scripts/eval_momentseeker.py --vlm-budgets 5 --rerank-frames 1  --out runs/ms_t1.json
$P -u scripts/eval_momentseeker.py --vlm-budgets 5 --rerank-frames 4  --out runs/ms_t4.json
echo ALLDONE
