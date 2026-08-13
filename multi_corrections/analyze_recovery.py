"""Can DWpose fill WiLoR's hand gaps? Over the 95k set, per frame:
 - WiLoR presence-change (a hand appeared/disappeared) vs DWpose presence-change, and overlap.
 - Per hand SIDE (L/R): present in WiLoR? present in DWpose? -> when WiLoR is MISSING a hand,
   can DWpose supply it (a usable bbox)?
Each shard writes aggregate counts to OUT/recovery_shard_<i>.json; summed afterwards.
"""
import os, sys, json, pickle, argparse
import numpy as np

VC="/home/ubuntu/us-west-3-fs/sahithi/Wilor/video_cut_wilor"
DW="/home/ubuntu/us-west-3-fs/live_dealer_blackjack/dwpose/batch_01/dwpose"
OUT="/home/ubuntu/us-west-3-fs/sahithi/hand_consistency/recovery_shards"

def vh(h):
    h=np.asarray(h,float)
    return h if (h.shape==(21,2) and np.isfinite(h).all() and (np.ptp(h[:,0])+np.ptp(h[:,1]))>1e-4) else None
def wrist(p,j):
    b=p.get('bodies')
    if b is None: return None
    c=np.asarray(b['candidate']); s=np.asarray(b['subset'])
    if s.ndim!=2: return None
    i=int(s[0,j])
    if 0<=i<len(c) and np.isfinite(c[i]).all(): return np.asarray(c[i],float)
    return None
def sides_key(pose, derive):
    """return (set_of_sides_present, presence_key). sides use L/R labels; key for presence-change."""
    if not pose or pose.get('hands') is None: return set(), frozenset()
    raw=np.asarray(pose['hands']); ir=pose.get('hands_is_right'); ir=np.asarray(ir) if ir is not None else None
    rw=lw=None
    if ir is None and derive: rw=wrist(pose,4); lw=wrist(pose,7)
    sides=set(); labs=[]
    for i,x in enumerate(raw):
        h=vh(x)
        if h is None: continue
        if ir is not None and i<len(ir): lab=bool(ir[i])
        elif rw is not None and lw is not None: lab=bool(np.linalg.norm(h[0]-rw)<np.linalg.norm(h[0]-lw))
        else: lab=None
        labs.append(lab)
        if lab is not None: sides.add(lab)
    key = frozenset(labs) if (labs and all(l is not None for l in labs) and len(set(labs))==len(labs)) else ("n",len(labs))
    return sides, key

def process(cid, agg):
    pw=f"{VC}/{cid}_dwpose.pkl"; pd=f"{DW}/{cid}_dwpose.pkl"
    if not (os.path.exists(pw) and os.path.exists(pd)): return
    try: W=pickle.load(open(pw,'rb')); D=pickle.load(open(pd,'rb'))
    except Exception: return
    n=min(len(W),len(D)); agg['frames']+=n
    pwk=pdk=None
    for k in range(n):
        ws,wk=sides_key(W[k].get('pose') if isinstance(W[k],dict) else None, derive=False)
        ds,dk=sides_key(D[k].get('pose') if isinstance(D[k],dict) else None, derive=True)
        wpc = (k>0 and pwk is not None and wk!=pwk)
        dpc = (k>0 and pdk is not None and dk!=pdk)
        if wpc: agg['wilor_pc']+=1
        if dpc: agg['dwpose_pc']+=1
        if wpc and dpc: agg['overlap_pc']+=1
        if wpc and not dpc: agg['wilor_pc_dwpose_ok']+=1
        # per-side missing-hand recovery
        for side in (True,False):
            if side not in ws:                       # WiLoR missing this hand
                agg['wilor_missing_side']+=1
                if side in ds: agg['recoverable_bbox']+=1   # DWpose has it
                else: agg['both_missing']+=1
        pwk,pdk=wk,dk

def new_agg(): return dict(frames=0,wilor_pc=0,dwpose_pc=0,overlap_pc=0,wilor_pc_dwpose_ok=0,
                           wilor_missing_side=0,recoverable_bbox=0,both_missing=0)

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument('--shard',type=int,default=0); ap.add_argument('--nshards',type=int,default=1)
    a=ap.parse_args(); os.makedirs(OUT,exist_ok=True)
    ids=sorted(f[:-len('_dwpose.pkl')] for f in os.listdir(VC) if f.endswith('_dwpose.pkl'))
    mine=[c for i,c in enumerate(ids) if i%a.nshards==a.shard]
    agg=new_agg()
    print(f"recovery shard {a.shard}/{a.nshards}: {len(mine)} clips",flush=True)
    for j,cid in enumerate(mine):
        try: process(cid, agg)
        except Exception: pass
        if (j+1)%2000==0: print(f"  shard{a.shard}: {j+1}/{len(mine)}",flush=True)
    json.dump(agg, open(f"{OUT}/recovery_shard_{a.shard}.json","w"))
    print(f"RECOVERY_SHARD_{a.shard}_DONE",flush=True)
