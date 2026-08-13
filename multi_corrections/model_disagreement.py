"""Model disagreement (hand): DWpose vs WiLoR.

Same clip/frame, both trackers give hand keypoints. For each hand SIDE (L/R) present in
BOTH, disagreement = mean over 21 keypoints of ||wilor_kpt - dwpose_kpt|| / hand_size.
Frame value = median over matched sides. When the two models place the same hand far
apart (> THRESH hand-widths), one of them is wrong.
  WiLoR L/R  : from hands_is_right
  DWpose L/R : derived by matching each hand root to body wrists (4=R,7=L)
Only frames where a side is present in BOTH are compared (missing-in-one = presence, not
disagreement). Bad frame = disagreement > THRESH; segment >= 2 consecutive.

Per clip -> OUT/<cid>.json:
  {clip, n_frames, n_compared, disagree_p50, disagree_p95, disagree_max, n_high, segments}
Sharded, resume-safe, same 95k set as the temporal checks.
"""
import os, sys, json, pickle, argparse
import numpy as np

VC="/home/ubuntu/us-west-3-fs/sahithi/Wilor/video_cut_wilor"          # WiLoR hands
DW="/home/ubuntu/us-west-3-fs/live_dealer_blackjack/dwpose/batch_01/dwpose"  # DWpose hands + body wrists
OUT="/home/ubuntu/us-west-3-fs/sahithi/hand_consistency/model_disagreement"
FPS=30.0; THRESH=0.5; MIN_RUN=2

def vh(h):
    h=np.asarray(h,float)
    return h if (h.shape==(21,2) and np.isfinite(h).all() and (np.ptp(h[:,0])+np.ptp(h[:,1]))>1e-4) else None
def hsz(h): return max(np.ptp(h[:,0]),np.ptp(h[:,1])) or 1e-3
def wrist(p,j):
    b=p.get('bodies')
    if b is None: return None
    c=np.asarray(b['candidate']); s=np.asarray(b['subset'])
    if s.ndim!=2: return None
    i=int(s[0,j])
    if 0<=i<len(c) and np.isfinite(c[i]).all(): return np.asarray(c[i],float)
    return None

def sided(pose, derive):
    """return {True:hand(R), False:hand(L)} for valid, uniquely-labelled hands."""
    if not pose or pose.get('hands') is None: return {}
    raw=np.asarray(pose['hands']); ir=pose.get('hands_is_right'); ir=np.asarray(ir) if ir is not None else None
    rw=lw=None
    if ir is None and derive:
        rw=wrist(pose,4); lw=wrist(pose,7)
    out={}; labs=[]
    hands=[]
    for i,x in enumerate(raw):
        h=vh(x)
        if h is None: continue
        if ir is not None and i<len(ir): lab=bool(ir[i])
        elif rw is not None and lw is not None: lab=bool(np.linalg.norm(h[0]-rw)<np.linalg.norm(h[0]-lw))
        else: lab=None
        hands.append(h); labs.append(lab)
    for h,l in zip(hands,labs):
        if l is None or l in out: return {k:v for k,v in out.items()} if len(set(labs))==len(labs) else {}  # ambiguous -> drop
        out[l]=h
    return out

def segs(mask):
    out=[]; s=None
    for i,v in enumerate(list(mask)+[False]):
        if v and s is None: s=i
        elif not v and s is not None:
            if (i-s)>=MIN_RUN: out.append([s,i-1,round(s/FPS,2),round(i/FPS,2)])
            s=None
    return out

def process(cid):
    pw=f"{VC}/{cid}_dwpose.pkl"; pd=f"{DW}/{cid}_dwpose.pkl"
    if not (os.path.exists(pw) and os.path.exists(pd)): return None
    try: W=pickle.load(open(pw,'rb')); D=pickle.load(open(pd,'rb'))
    except Exception: return None
    n=min(len(W),len(D)); vals=[]; mask=[False]*n; ncmp=0
    for k in range(n):
        wh=sided(W[k].get('pose') if isinstance(W[k],dict) else None, derive=False)
        dh=sided(D[k].get('pose') if isinstance(D[k],dict) else None, derive=True)
        per=[]
        for side in (True,False):
            if side in wh and side in dh:
                a,b=wh[side],dh[side]; sc=(hsz(a)+hsz(b))/2
                per.append(float((np.linalg.norm(a-b,axis=1)/sc).mean()))
        if per:
            ncmp+=1; v=float(np.median(per)); vals.append(v); mask[k]=v>THRESH
    def pct(q): return round(float(np.percentile(vals,q)),4) if vals else None
    rec=dict(clip=cid, n_frames=n, n_compared=ncmp,
             disagree_p50=pct(50), disagree_p95=pct(95),
             disagree_max=round(float(max(vals)),4) if vals else None,
             n_high=int(sum(mask)), segments=segs(mask))
    json.dump(rec, open(f"{OUT}/{cid}.json","w"))
    return rec

def all_ids():
    return sorted(f[:-len('_dwpose.pkl')] for f in os.listdir(VC) if f.endswith('_dwpose.pkl'))

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument('--shard',type=int,default=0); ap.add_argument('--nshards',type=int,default=1)
    a=ap.parse_args(); os.makedirs(OUT,exist_ok=True)
    ids=all_ids(); mine=[c for i,c in enumerate(ids) if i%a.nshards==a.shard]
    print(f"disagreement shard {a.shard}/{a.nshards}: {len(mine)} clips",flush=True)
    done=0
    for cid in mine:
        try: process(cid)
        except Exception: pass
        done+=1
        if done%2000==0: print(f"  shard{a.shard}: {done}/{len(mine)}",flush=True)
    print(f"DISAGREE_SHARD_{a.shard}_DONE",flush=True)
