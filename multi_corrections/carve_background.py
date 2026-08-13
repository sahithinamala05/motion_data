"""Carve 'background' in livedealer_temporal_metatext_all_actions so no frame is
labeled background if any real action covers it. Foreground = every non-background
segment. Each background segment is trimmed by subtracting the union of foreground
frame-intervals; residual pieces stay as background. Foreground segments untouched.
In-place rewrite, same schema. Idempotent."""
import os, json
from multiprocessing import Pool

D="/home/ubuntu/us-west-3-fs/sahithi/livedealer_temporal_metatext_all_actions"

def merge_intervals(ivs):
    ivs=sorted(ivs); out=[]
    for a,b in ivs:
        if out and a<=out[-1][1]+1: out[-1][1]=max(out[-1][1],b)
        else: out.append([a,b])
    return out

def subtract(a,b, occ):
    """return list of [s,e] sub-intervals of [a,b] not covered by occ (merged, sorted)."""
    res=[]; cur=a
    for oa,ob in occ:
        if ob<cur or oa>b: continue
        if oa>cur: res.append([cur, min(oa-1,b)])
        cur=max(cur, ob+1)
        if cur>b: break
    if cur<=b: res.append([cur,b])
    return [r for r in res if r[0]<=r[1]]

def bg_seg(a,b):
    return {"start_frame":a,"end_frame":b,"labels":["background"],"meta_text":[],
            "duration_frames":b-a+1,"id":"","bounding_boxes":[]}

def carve(path):
    try: rec=json.load(open(path))
    except Exception: return (0,0)
    segs=rec.get("timeline_segments",[])
    fg=[s for s in segs if s.get("labels",["background"])[:1]!=["background"]]
    bg=[s for s in segs if s.get("labels",["background"])[:1]==["background"]]
    if not bg: return (0,0)
    occ=merge_intervals([[s["start_frame"],s["end_frame"]] for s in fg]) if fg else []
    new_bg=[]
    for s in bg:
        for a,b in (subtract(s["start_frame"],s["end_frame"],occ) if occ else [[s["start_frame"],s["end_frame"]]]):
            new_bg.append(bg_seg(a,b))
    out=fg+new_bg
    out.sort(key=lambda s:(s["start_frame"],s["end_frame"]))
    rec["timeline_segments"]=out
    json.dump(rec, open(path,"w"))
    return (len(bg), len(new_bg))

if __name__=="__main__":
    files=[os.path.join(D,f) for f in os.listdir(D) if f.endswith("_predictions.json")]
    print(f"carving background in {len(files)} files ...",flush=True)
    bg_before=bg_after=0
    with Pool(24) as pool:
        for i,(a,b) in enumerate(pool.imap_unordered(carve, files, chunksize=64)):
            bg_before+=a; bg_after+=b
            if (i+1)%5000==0: print(f"  {i+1}/{len(files)}",flush=True)
    print(f"DONE. background segments: {bg_before} -> {bg_after} (split/trimmed around actions)")
