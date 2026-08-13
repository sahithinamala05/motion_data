"""Temporal inconsistency (||p_t - p_{t-1}||, scale-normalized) for WiLoR hands and DWpose body.

WiLoR hands: video_cut_wilor/<cid>_dwpose.pkl  pose.hands (21,2) / hand_size  -> threshold 0.5
DWpose body: dwpose/batch_01/dwpose/<cid>_dwpose.pkl  wrists(4,7)+elbows(3,6) / shoulder_width -> 0.3
Per frame value = median over tracked items of displacement vs previous frame.
Bad frame = value>thresh; bad segment = >=2 consecutive (MIN_RUN=2).

Per clip -> OUT/<cid>.json:
  {clip, n_frames,
   wilor:{p50,p95,max,n_bad,segments},
   body :{p50,p95,max,n_bad,segments}}
Sharded, resume-safe, pkl-only (no video/blur).
"""
import os, sys, json, pickle, argparse
import numpy as np

VC="/home/ubuntu/us-west-3-fs/sahithi/Wilor/video_cut_wilor"          # WiLoR hands (95k packed set)
DW="/home/ubuntu/us-west-3-fs/live_dealer_blackjack/dwpose/batch_01/dwpose"  # body
OUT_W="/home/ubuntu/us-west-3-fs/sahithi/hand_consistency/wilor_temporal"
OUT_B="/home/ubuntu/us-west-3-fs/sahithi/hand_consistency/body_temporal"
FPS=30.0; TH_HAND=0.5; TH_BODY=0.3; MIN_RUN=2
RWRI,LWRI,RELB,LELB,RSHO,LSHO=4,7,3,6,2,5

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
def hand_val(cur,prev):
    if not cur or not prev: return np.nan
    v=[np.mean(np.linalg.norm(a-b,axis=1))/((handsize(a)+handsize(b))/2) for a,b in match(cur,prev)]
    return float(np.median(v)) if v else np.nan
def body_pt(c,s,j):
    i=int(s[0,j])
    if i<0 or i>=len(c): return None
    x,y=c[i]; return np.array([x,y]) if np.isfinite(x) and np.isfinite(y) else None
def segs(mask):
    out=[]; s=None
    for i,v in enumerate(list(mask)+[False]):
        if v and s is None: s=i
        elif not v and s is not None:
            if (i-s)>=MIN_RUN: out.append([s,i-1,round(s/FPS,2),round(i/FPS,2)])
            s=None
    return out
def summ(vals,mask,n):
    def pct(q): return round(float(np.percentile(vals,q)),4) if vals else None
    return dict(p50=pct(50),p95=pct(95),max=round(float(max(vals)),4) if vals else None,
                n_bad=int(sum(mask)),segments=segs(mask))

def process(cid):
    pw=f"{VC}/{cid}_dwpose.pkl"; pd=f"{DW}/{cid}_dwpose.pkl"
    if not (os.path.exists(pw) and os.path.exists(pd)): return None
    try: W=pickle.load(open(pw,'rb')); D=pickle.load(open(pd,'rb'))
    except Exception: return None
    n=min(len(W),len(D))
    wv=[]; wm=[False]*n; bv=[]; bm=[False]*n; prevW=None; prevB={}
    for k in range(n):
        pWa=W[k].get('pose') if isinstance(W[k],dict) else None
        pDa=D[k].get('pose') if isinstance(D[k],dict) else None
        Wh=[h for h in [valid_hand(x) for x in np.asarray(pWa.get('hands'))] if h is not None] if (pWa and pWa.get('hands') is not None) else []
        hv=hand_val(Wh,prevW)
        if np.isfinite(hv): wv.append(hv); wm[k]=hv>TH_HAND
        # body
        curB={}
        if pDa and pDa.get('bodies') is not None:
            c=np.asarray(pDa['bodies']['candidate']); s=np.asarray(pDa['bodies']['subset'])
            rs,ls=body_pt(c,s,RSHO),body_pt(c,s,LSHO)
            sw=np.linalg.norm(rs-ls) if (rs is not None and ls is not None) else np.nan
            curB={j:body_pt(c,s,j) for j in (RWRI,LWRI,RELB,LELB)}; curB={j:p for j,p in curB.items() if p is not None}
            bd=[np.linalg.norm(curB[j]-prevB[j])/sw for j in curB if j in prevB] if (np.isfinite(sw) and sw>0) else []
            if bd:
                val=float(np.median(bd)); bv.append(val); bm[k]=val>TH_BODY
        prevW=Wh; prevB=curB
    recW=dict(clip=cid,n_frames=n,**summ(wv,wm,n)); recW['source']='wilor_hand'
    recB=dict(clip=cid,n_frames=n,**summ(bv,bm,n)); recB['source']='dwpose_body'
    json.dump(recW, open(f"{OUT_W}/{cid}.json","w"))
    json.dump(recB, open(f"{OUT_B}/{cid}.json","w"))
    return recW,recB

def all_ids():
    return sorted(f[:-len('_dwpose.pkl')] for f in os.listdir(VC) if f.endswith('_dwpose.pkl'))

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument('--shard',type=int,default=0); ap.add_argument('--nshards',type=int,default=1)
    a=ap.parse_args(); os.makedirs(OUT_W,exist_ok=True); os.makedirs(OUT_B,exist_ok=True)
    ids=all_ids(); mine=[c for i,c in enumerate(ids) if i%a.nshards==a.shard]
    print(f"shard {a.shard}/{a.nshards}: {len(mine)} clips (of {len(ids)})",flush=True)
    done=0
    for cid in mine:
        if os.path.exists(f"{OUT_W}/{cid}.json") and os.path.exists(f"{OUT_B}/{cid}.json"): done+=1; continue
        try: process(cid)
        except Exception: pass
        done+=1
        if done%2000==0: print(f"  shard{a.shard}: {done}/{len(mine)}",flush=True)
    print(f"SHARD_{a.shard}_DONE",flush=True)
