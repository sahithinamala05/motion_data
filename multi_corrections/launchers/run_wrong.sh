#!/bin/bash
set +e
SP=/tmp/claude-1000/-home-ubuntu/88cc26b9-b104-4dd1-b365-b88dd225bc5c/scratchpad
VW=/home/ubuntu/us-west-3-fs/sahithi/multi_correction/vis_wrong
PY=/home/ubuntu/miniconda3/envs/wilor/bin/python
rm -rf "$VW"; mkdir -p "$VW"
# 2 workers per list
for s in 0 1; do
  OMP_NUM_THREADS=3 "$PY" "$SP/runner_viz.py" "$SP/case_lobby.txt" "$VW" "$s" 2 > "$SP/w_lobby_$s.log" 2>&1 &
  OMP_NUM_THREADS=3 "$PY" "$SP/runner_viz.py" "$SP/case_table.txt" "$VW" "$s" 2 > "$SP/w_table_$s.log" 2>&1 &
done
wait
# prefix filenames so the two cases are distinguishable
while read st; do [ -n "$st" ] && mv "$VW/${st}_mid.mp4" "$VW/lobby_${st}.mp4" 2>/dev/null; done < "$SP/case_lobby.txt"
while read st; do [ -n "$st" ] && mv "$VW/${st}_mid.mp4" "$VW/table_${st}.mp4" 2>/dev/null; done < "$SP/case_table.txt"
echo "DONE"
echo "lobby_ : $(ls "$VW"/lobby_*.mp4 2>/dev/null | wc -l)   table_ : $(ls "$VW"/table_*.mp4 2>/dev/null | wc -l)"
ls "$VW"
