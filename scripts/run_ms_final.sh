#!/usr/bin/env bash
# Final MomentSeeker sweep, with the top-k chunk aggregation that replaced max.
#
# The first two rows are the ablation for that one change, holding everything
# else fixed; the last two add the expensive stage on top of it.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
P=.venv/bin/python

$P -u scripts/eval_momentseeker.py --agg max                --vlm-budgets 0  --out runs/msf_max.json
$P -u scripts/eval_momentseeker.py --agg topk --topk 4      --vlm-budgets 0  --out runs/msf_topk.json
$P -u scripts/eval_momentseeker.py --agg topk --topk 4 --rerank-frames 4 --vlm-budgets 5  --out runs/msf_topk_c5.json
$P -u scripts/eval_momentseeker.py --agg topk --topk 4 --rerank-frames 4 --vlm-budgets 10 --out runs/msf_topk_c10.json
echo ALLDONE
