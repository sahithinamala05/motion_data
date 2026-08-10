#!/bin/bash
set +e
cd /home/ubuntu/us-west-3-fs/sahithi/_entire_code/multi_corrections/
BASE=/home/ubuntu/us-west-3-fs/sahithi/multi_correction/full_run
SP=/tmp/claude-1000/-home-ubuntu/88cc26b9-b104-4dd1-b365-b88dd225bc5c/scratchpad
PY=/home/ubuntu/miniconda3/envs/wilor/bin/python
rm -rf "$BASE/pkl" "$BASE/vis"
mkdir -p "$BASE/pkl" "$BASE/vis"
for s in $(seq 0 9); do
  OMP_NUM_THREADS=2 "$PY" dealer_faceid.py --shard "$s" --nshards 10 --viz_first 100 > "$SP/nt_shard_$s.log" 2>&1 &
done
wait
echo "ALL SHARDS DONE"
echo "pkls: $(find "$BASE/pkl" -name '*.pkl' | wc -l)  viz: $(find "$BASE/vis" -name '*.mp4' | wc -l)"
echo "errors: $(grep -h ERR "$SP"/nt_shard_*.log 2>/dev/null | wc -l)"
