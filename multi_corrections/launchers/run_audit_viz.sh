#!/bin/bash
set +e
SP=/tmp/claude-1000/-home-ubuntu/88cc26b9-b104-4dd1-b365-b88dd225bc5c/scratchpad
BASE=/home/ubuntu/us-west-3-fs/sahithi/multi_correction/full_run
PY=/home/ubuntu/miniconda3/envs/wilor/bin/python
rm -rf "$BASE/vis_wrist" "$BASE/vis_behind"
mkdir -p "$BASE/vis_wrist" "$BASE/vis_behind"
for s in 0 1 2 3 4; do
  OMP_NUM_THREADS=2 "$PY" "$SP/runner_viz.py" "$SP/wrist100.txt" "$BASE/vis_wrist"  "$s" 5 > "$SP/av_wrist_$s.log"  2>&1 &
  OMP_NUM_THREADS=2 "$PY" "$SP/runner_viz.py" "$SP/behind.txt"   "$BASE/vis_behind" "$s" 5 > "$SP/av_behind_$s.log" 2>&1 &
done
wait
echo "DONE"
echo "vis_wrist: $(find "$BASE/vis_wrist" -name '*.mp4'|wc -l)/100   vis_behind: $(find "$BASE/vis_behind" -name '*.mp4'|wc -l)/100"
echo "errors: $(grep -h ERR "$SP"/av_*.log 2>/dev/null|wc -l)"
