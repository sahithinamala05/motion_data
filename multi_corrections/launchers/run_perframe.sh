#!/bin/bash
# Per-frame dark+blur quality analysis over ALL remaining batch01 clips (94,004),
# 24 CPU shards, then merge -> rem_95k_filtered/{quality_perframe.parquet,
# quality_passed.jsonl/.txt, quality_summary.txt}. OCR folded in per-file at merge.
set +e
cd /home/ubuntu/us-west-3-fs/sahithi/_entire_code/multi_corrections/
SP=/tmp/quality_perframe_logs        # NOTE: shard logs go here; edit before reuse
PY=/home/ubuntu/miniconda3/envs/wilor/bin/python
N=24
mkdir -p "$SP"
for s in $(seq 0 $((N-1))); do
  OMP_NUM_THREADS=1 "$PY" filter_quality_perframe.py --shard "$s" --nshards "$N" > "$SP/pf_shard_$s.log" 2>&1 &
done
wait
echo "ALL SHARDS DONE"
"$PY" filter_quality_perframe.py --merge
