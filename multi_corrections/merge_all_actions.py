"""Build livedealer_temporal_metatext_all_actions/: per-video union of temporal segments.

Base           : livedealer_temporal_metatext_v2_index_coordinates/<id>_predictions.json
                 (initial-hand timeline: background, discard, initial hands 1st/2nd,
                  reveal hole card, close bets)
+ clean hand    : hard_action_annotations/full_run/clean_hand_verified/<id>_annotations.json
+ call for action: live_dealer_blackjack/action_annotation/call_for_action_2k5/<id>_annotations.json
+ hit/dealer hit: live_dealer_blackjack/action_annotation/hit_dealer_hit_2k5/<id>_annotations.json
+ split         : hard_action_annotations/full_run/split_sam_traj_all/trajectories_reformated/
                  <id>_trajectory.jsonl  (header line -> one 'split' segment)

Union (non-destructive): base segments kept; extra action segments appended, each tagged
with 'source'; segments sorted by start_frame. Videos present in an action source but not
in base are written with their action segments only. Output: <id>_predictions.json.
"""
import os, sys, json, glob
from multiprocessing import Pool

BASE="/home/ubuntu/us-west-3-fs/sahithi/livedealer_temporal_metatext_v2_index_coordinates"
OUT ="/home/ubuntu/us-west-3-fs/sahithi/livedealer_temporal_metatext_all_actions"
ANNO={  # name -> (dir, suffix)
 "clean_hand_verified": ("/home/ubuntu/us-west-3-fs/sahithi/hard_action_annotations/full_run/clean_hand_verified","_annotations.json"),
 "final_cfa":           ("/home/ubuntu/us-west-3-fs/sahithi/hard_action_annotations/final/cfa","_annotations.json"),
 "final_hit":           ("/home/ubuntu/us-west-3-fs/sahithi/hard_action_annotations/final/hit","_annotations.json"),
}
SPLIT=("/home/ubuntu/us-west-3-fs/sahithi/hard_action_annotations/full_run/split_sam_traj_all/trajectories_reformated","_trajectory.jsonl")

def load_index(d,suf):
    return {f[:-len(suf)]:os.path.join(d,f) for f in os.listdir(d) if f.endswith(suf)} if os.path.isdir(d) else {}

SEG_KEYS=("start_frame","end_frame","labels","meta_text","duration_frames","id","bounding_boxes")
def _clean(s):
    """keep exactly the base segment schema, in order."""
    return {k:s.get(k, [] if k in ("labels","meta_text","bounding_boxes") else "") for k in SEG_KEYS}

def anno_segments(path):
    try: j=json.load(open(path))
    except Exception: return []
    return [_clean(s) for s in j.get('timeline_segments',[])]

def split_segments(path):
    out=[]
    for line in open(path):
        line=line.strip()
        if not line: continue
        try: o=json.loads(line)
        except Exception: continue
        if o.get('type')=='header' and o.get('segment_start') is not None:
            a,b=int(o['segment_start']),int(o['segment_end'])
            out.append(_clean(dict(start_frame=a,end_frame=b,labels=["split"],
                            meta_text=[o.get('sub_bucket','')],duration_frames=b-a+1,
                            id="",bounding_boxes=[])))
    return out

# globals for workers
IDX={}
def build(vid):
    segs=[]
    base_p=IDX['base'].get(vid)
    if base_p:
        try: j=json.load(open(base_p))
        except Exception: j={"timeline_segments":[]}
        segs+=[dict(s) for s in j.get('timeline_segments',[])]   # keep base as-is (preserves xy_seat + any extra fields)
    for name in ("clean_hand_verified","final_cfa","final_hit"):
        p=IDX[name].get(vid)
        if p: segs+=anno_segments(p)
    sp=IDX['split'].get(vid)
    if sp: segs+=split_segments(sp)
    segs.sort(key=lambda s:(s.get('start_frame',0),s.get('end_frame',0)))
    # exact base schema: video_name, video_id, video_path, timeline_segments
    rec={"video_name":f"{vid}.mp4","video_id":"","video_path":"","timeline_segments":segs}
    json.dump(rec, open(f"{OUT}/{vid}_predictions.json","w"))
    # return which action labels this video ended up with (for stats)
    labs=set()
    for s in segs:
        for l in s.get('labels',[]): labs.add(l)
    return labs

def _init(idx):
    global IDX; IDX=idx

if __name__=="__main__":
    os.makedirs(OUT,exist_ok=True)
    idx={'base':load_index(BASE,"_predictions.json")}
    for name,(d,suf) in ANNO.items(): idx[name]=load_index(d,suf)
    idx['split']=load_index(*SPLIT)
    # universe = base ids UNION all source ids
    vids=set(idx['base'])
    for k in list(ANNO)+['split']: vids|=set(idx[k])
    vids=sorted(vids)
    print(f"videos to write: {len(vids)}  (base {len(idx['base'])}, "
          f"+cha {len(idx['clean_hand_verified'])}, +cfa {len(idx['final_cfa'])}, "
          f"+hit {len(idx['final_hit'])}, +split {len(idx['split'])})", flush=True)
    import collections
    lab_files=collections.Counter()
    with Pool(24, initializer=_init, initargs=(idx,)) as pool:
        for i,labs in enumerate(pool.imap_unordered(build, vids, chunksize=64)):
            for l in labs: lab_files[l]+=1
            if (i+1)%5000==0: print(f"  {i+1}/{len(vids)}", flush=True)
    print(f"\nDONE wrote {len(vids)} -> {OUT}")
    print("videos containing each action label:")
    for l,c in lab_files.most_common(): print(f"  {c:6d}  {l}")
