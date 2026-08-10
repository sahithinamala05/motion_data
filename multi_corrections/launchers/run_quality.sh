#!/bin/bash
# Robust launcher: each shard is a fully-detached (setsid+nohup) process that survives session teardown.
# Resumable — re-running skips clips already in videos/parts/part_*.jsonl. Merge separately when done.
set +e
cd /home/ubuntu/us-west-3-fs/sahithi/_entire_code/multi_corrections/
SP=/tmp/claude-1000/-home-ubuntu/88cc26b9-b104-4dd1-b365-b88dd225bc5c/scratchpad
PY=/home/ubuntu/miniconda3/envs/wilor/bin/python
mkdir -p /home/ubuntu/us-west-3-fs/sahithi/multi_correction/videos/parts
N=20
for s in $(seq 0 $((N-1))); do
  OMP_NUM_THREADS=1 setsid nohup "$PY" filter_quality.py --shard "$s" --nshards "$N" > "$SP/q_shard_$s.log" 2>&1 < /dev/null &
done
echo "launched $N detached shards"
