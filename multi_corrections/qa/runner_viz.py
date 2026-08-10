import sys, os
sys.path.insert(0, "/home/ubuntu/us-west-3-fs/sahithi/_entire_code/multi_corrections")
import dealer_faceid as m
stems_file, vizdir, shard, nsh = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
m.VIS_DIR = vizdir
os.makedirs(vizdir, exist_ok=True)
stems=[l.strip() for l in open(stems_file) if l.strip()]
mine=[s for i,s in enumerate(stems) if i%nsh==shard]
ok=0
for s in mine:
    try:
        m.process_clip(s, viz=True); ok+=1
    except Exception as e:
        print("ERR", s, e, flush=True)
print(f"shard {shard} done {ok}/{len(mine)}", flush=True)
