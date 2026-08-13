"""DWpose hand-pose temporal inconsistency:  value_t = ||p_t - p_{t-1}|| (scale-normalized).

Source: live_dealer_blackjack/dwpose/batch_01/dwpose/<cid>_dwpose.pkl  (pose.hands, (K,21,2)).
Per frame with hands, match each hand to the nearest hand in the previous frame (by root
keypoint) and compute mean over 21 keypoints of ||p_t - p_{t-1}|| / hand_size; the frame's
value = median over matched hands. A frame is temporally-inconsistent if value > THRESH
(0.5 hand-widths). Consecutive bad frames -> segments (isolated 1-frame drops; MIN_RUN=2).

Per clip -> OUT/<cid>.json:
  {clip, n_frames, n_hand_frames, temp_p50, temp_p95, temp_max, n_bad, segments}
segments = [[start_f,end_f,start_s,end_s], ...].  pkl-only (no video), sharded, resume-safe.
"""
import os, sys, json, pickle, argparse
import numpy as np

DW="/home/ubuntu/us-west-3-fs/live_dealer_blackjack/dwpose/batch_01/dwpose"
OUT="/home/ubuntu/us-west-3-fs/sahithi/hand_consistency/dwpose_temporal"
FPS=30.0; THRESH=0.5; MIN_RUN=2

def valid_hand(h):
    h=np.asarray(h,float)
    if h.ndim!=2 or h.shape!=(21,2): return None
    return h if (np.isfinite(h).all() and (np.ptp(h[:,0])+np.ptp(h[:,1]))>1e-4) else None
def handsize(h): return max(np.ptp(h[:,0]),np.ptp(h[:,1])) or 1e-3
def match(cur,prev):
    pairs=[]; used=set()
    for a in cur:
        best=None;bd=1e9
        for i,b in enumerate(prev):
            if i in used: continue
            d=np.linalg.norm(a[0]-b[0])
            if d<bd: bd=d;best=i
        if best is not None: used.add(best); pairs.append((a,prev[best]))
    return pairs
def frame_val(cur,prev):
    if not cur or not prev: return np.nan
    v=[np.mean(np.linalg.norm(a-b,axis=1))/((handsize(a)+handsize(b))/2) for a,b in match(cur,prev)]
    return float(np.median(v)) if v else np.nan
def segs(mask):
    out=[]; s=None
    for i,v in enumerate(list(mask)+[False]):
        if v and s is None: s=i
        elif not v and s is not None:
            if (i-s)>=MIN_RUN: out.append([s,i-1,round(s/FPS,2),round(i/FPS,2)])
            s=None
    return out

def process(cid):
    try: D=pickle.load(open(f"{DW}/{cid}_dwpose.pkl",'rb'))
    except Exception: return None
    n=len(D); vals=[]; mask=[False]*n; prev=None; nhand=0
    for k in range(n):
        p=D[k].get('pose') if isinstance(D[k],dict) else None
        hs=[h for h in [valid_hand(x) for x in np.asarray(p.get('hands'))] if h is not None] if (p and p.get('hands') is not None) else []
        if hs: nhand+=1
        val=frame_val(hs,prev)
        if np.isfinite(val):
            vals.append(val); mask[k]= val>THRESH
        prev=hs
    def pct(a,q): return round(float(np.percentile(a,q)),4) if a else None
    rec=dict(clip=cid, n_frames=n, n_hand_frames=nhand,
             temp_p50=pct(vals,50), temp_p95=pct(vals,95),
             temp_max=round(float(max(vals)),4) if vals else None,
             n_bad=int(sum(mask)), segments=segs(mask))
    json.dump(rec, open(f"{OUT}/{cid}.json","w"))
    return rec

def all_ids(subset_dir=None, subset_suf='_dwpose.pkl'):
    ids=set(f[:-len('_dwpose.pkl')] for f in os.listdir(DW) if f.endswith('_dwpose.pkl'))
    if subset_dir:
        sub=set(f[:-len(subset_suf)] for f in os.listdir(subset_dir) if f.endswith(subset_suf))
        ids &= sub
    return sorted(ids)

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument('--shard',type=int,default=0); ap.add_argument('--nshards',type=int,default=1)
    ap.add_argument('--subset_dir',type=str,default=None)
    a=ap.parse_args(); os.makedirs(OUT,exist_ok=True)
    ids=all_ids(a.subset_dir); mine=[c for i,c in enumerate(ids) if i%a.nshards==a.shard]
    print(f"shard {a.shard}/{a.nshards}: {len(mine)} clips (of {len(ids)})",flush=True)
    done=0
    for cid in mine:
        if os.path.exists(f"{OUT}/{cid}.json"): done+=1; continue
        try: process(cid)
        except Exception: pass
        done+=1
        if done%2000==0: print(f"  shard{a.shard}: {done}/{len(mine)}",flush=True)
    print(f"SHARD_{a.shard}_DONE",flush=True)
