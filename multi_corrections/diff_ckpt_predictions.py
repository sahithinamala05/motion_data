"""Diff FACT initial-hand predictions between an OLD checkpoint (iter-40000) and the
LATEST output we already have (livedealer_temporal_metatext_v2_index_coordinates).

Both are per-frame semantic segmentations (non-overlapping timeline_segments). For each
common clip we expand to per-frame label arrays, align on min length, and compute:
  n_frames, n_diff, frac_diff, and the frame-level (old->new) label changes.
Writes per-clip diff JSON to OUT/diff/<cid>.json and a global OUT/summary.{json,txt}
(overall frac differing, per-class agreement, global old->new change matrix, worst clips).
"""
import os, sys, json, collections
import numpy as np
from multiprocessing import Pool

OLD="/home/ubuntu/us-west-3-fs/sahithi/initial_hand_ckpt_diff/pred_40000/predictions_json"   # iter-40000
NEW="/home/ubuntu/us-west-3-fs/sahithi/livedealer_temporal_metatext_v2_index_coordinates"     # latest (iter-64000-ish)
OUT="/home/ubuntu/us-west-3-fs/sahithi/initial_hand_ckpt_diff"
DIFFD=os.path.join(OUT,"diff")

def frames(path, suf):
    j=json.load(open(path)); segs=j.get("timeline_segments",[])
    if not segs: return np.array([],dtype=object)
    N=max(s["end_frame"] for s in segs)+1
    a=np.array(["background"]*N, dtype=object)
    for s in segs:
        lab=s["labels"][0] if s["labels"] else "background"
        a[s["start_frame"]:s["end_frame"]+1]=lab
    return a

def one(cid):
    try:
        A=frames(f"{OLD}/{cid}.json","")          # old iter-40000
        B=frames(f"{NEW}/{cid}_predictions.json","")  # latest
    except Exception:
        return None
    n=min(len(A),len(B))
    if n==0: return None
    A=A[:n]; B=B[:n]
    diff_mask=A!=B
    nd=int(diff_mask.sum())
    changes=collections.Counter(zip(A[diff_mask], B[diff_mask]))
    rec=dict(clip=cid, n_frames=n, n_diff=nd, frac_diff=round(nd/n,4),
             changes=[{"from":f,"to":t,"frames":int(c)} for (f,t),c in changes.most_common()])
    json.dump(rec, open(f"{DIFFD}/{cid}.json","w"))
    # return compact stats for aggregation
    return (cid, n, nd, [(f,t,int(c)) for (f,t),c in changes.items()])

if __name__=="__main__":
    os.makedirs(DIFFD, exist_ok=True)
    old_ids=set(f[:-5] for f in os.listdir(OLD) if f.endswith(".json"))
    new_ids=set(f[:-len("_predictions.json")] for f in os.listdir(NEW) if f.endswith("_predictions.json"))
    common=sorted(old_ids & new_ids)
    print(f"old(iter40000)={len(old_ids)} new(latest)={len(new_ids)} common={len(common)}",flush=True)
    tot_f=tot_d=0; per_clip=[]; global_ch=collections.Counter()
    with Pool(24) as pool:
        for r in pool.imap_unordered(one, common, chunksize=64):
            if not r: continue
            cid,n,nd,ch=r; tot_f+=n; tot_d+=nd; per_clip.append((cid,n,nd,nd/n))
            for f,t,c in ch: global_ch[(f,t)]+=c
    fracs=np.array([p[3] for p in per_clip])
    summary=dict(
        clips=len(per_clip), total_frames=tot_f, total_diff_frames=tot_d,
        overall_frac_diff=round(tot_d/max(tot_f,1),4),
        per_clip_frac_diff=dict(mean=round(float(fracs.mean()),4), median=round(float(np.median(fracs)),4),
                                p90=round(float(np.percentile(fracs,90)),4), max=round(float(fracs.max()),4)),
        clips_ge_5pct=int((fracs>=0.05).sum()), clips_ge_10pct=int((fracs>=0.10).sum()),
        top_change_transitions=[{"from":f,"to":t,"frames":int(c)} for (f,t),c in global_ch.most_common(25)],
    )
    json.dump(summary, open(f"{OUT}/summary.json","w"), indent=1)
    with open(f"{OUT}/summary.txt","w") as fo:
        fo.write("FACT initial-hand ckpt diff: iter-40000 (old) vs latest v2 output\n")
        fo.write(f"clips compared      : {summary['clips']}\n")
        fo.write(f"overall frame diff  : {tot_d}/{tot_f} = {100*summary['overall_frac_diff']:.2f}%\n")
        fo.write(f"per-clip frac_diff  : mean {summary['per_clip_frac_diff']['mean']*100:.2f}%  "
                 f"median {summary['per_clip_frac_diff']['median']*100:.2f}%  "
                 f"p90 {summary['per_clip_frac_diff']['p90']*100:.2f}%  max {summary['per_clip_frac_diff']['max']*100:.2f}%\n")
        fo.write(f"clips >=5% changed  : {summary['clips_ge_5pct']}   >=10%: {summary['clips_ge_10pct']}\n\n")
        fo.write("top old->new label transitions (frames):\n")
        for d in summary['top_change_transitions']:
            fo.write(f"  {d['frames']:8d}  {d['from']:>18s}  ->  {d['to']}\n")
    print(open(f"{OUT}/summary.txt").read())
    print(f"per-clip diffs -> {DIFFD}")
