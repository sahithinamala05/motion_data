#!/bin/bash
# Apply the finalised per-frame quality result to ALL single-dealer pkls:
# null pose on dark/blur frames -> full_run/pkl_updated/. Fast (pkl edits, no decode).
# RUN ONLY AFTER quality_perframe.parquet is finalised (the blur check has merged).
# Then optionally render a comparison-viz sample.
set +e
cd /home/ubuntu/us-west-3-fs/sahithi/_entire_code/multi_corrections/
SP=/tmp/apply_pkl_logs          # shard logs; edit before reuse
PY=/home/ubuntu/miniconda3/envs/wilor/bin/python
N=16
mkdir -p "$SP"
for s in $(seq 0 $((N-1))); do
  "$PY" apply_quality_to_pkl.py --shard "$s" --nshards "$N" > "$SP/apply_$s.log" 2>&1 &
done
wait
echo "ALL PKLS UPDATED -> full_run/pkl_updated/"
echo "rendering comparison viz sample (20, left OLD | right NEW)"
"$PY" apply_quality_to_pkl.py --viz --nviz 20
echo "DONE"
