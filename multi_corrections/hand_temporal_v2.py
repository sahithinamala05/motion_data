"""Hand temporal check v2 = jump (p_t - p_{t-1}) + PRESENCE change.

--source {dwpose, wilor}. Per frame:
  * valid hands + is_right labels.
  * JUMP: match hands to previous frame by is_right label (L<->L, R<->R; fall back to
    nearest-root if labels missing/duplicate). jump = median over matched hands of
    mean_kpt||p_t-p_{t-1}|| / handsize. Matching by label avoids the fake giant jumps that
    nearest-root produced when hand count changed.
  * PRESENCE: the SET of hands present changed vs previous frame (a hand appeared,
    disappeared, or L/R flipped). Uses is_right sets when available, else hand count.
A frame is BAD if jump>THRESH (0.5) OR presence changed.
  jump_segments : consecutive jump-bad frames (>=2, isolated blips ignored)
  presence_frames: every frame where presence changed (transitions; no min-run)
  bad_segments  : consecutive BAD (jump OR presence) frames (>=1)

Per clip -> OUT[source]/<cid>.json. Sharded, overwrite (schema changed).
"""
import os, sys, json, pickle, argparse
import numpy as np

VC="/home/ubuntu/us-west-3-fs/sahithi/Wilor/video_cut_wilor"
DW="/home/ubuntu/us-west-3-fs/live_dealer_blackjack/dwpose/batch_01/dwpose"
OUT={"wilor":"/home/ubuntu/us-west-3-fs/sahithi/hand_consistency/wilor_temporal",
     "dwpose":"/home/ubuntu/us-west-3-fs/sahithi/hand_consistency/dwpose_temporal"}
SRCDIR={"wilor":VC,"dwpose":DW}
FPS=30.0; THRESH=1.0; MIN_RUN=2   # bad jump = moved more than a full hand-width in one frame

def vh(h):
    h=np.asarray(h,float)
    return h if (h.shape==(21,2) and np.isfinite(h).all() and (np.ptp(h[:,0])+np.ptp(h[:,1]))>1e-4) else None
def hsz(h): return max(np.ptp(h[:,0]),np.ptp(h[:,1])) or 1e-3

def _wrist(p,j):
    b=p.get('bodies')
    if b is None: return None
    c=np.asarray(b['candidate']); s=np.asarray(b['subset'])
    if s.ndim!=2: return None
    i=int(s[0,j])
    if 0<=i<len(c):
        x,y=c[i]
        if np.isfinite(x) and np.isfinite(y): return np.array([x,y])
    return None

def get_hands(fr):
    """return (hands[list of (21,2)], labels[list of True(R)/False(L)/None], presence_key).
    Labels from hands_is_right if present (WiLoR), else derived by matching each hand's
    root to the body wrists R(4)/L(7) (DWpose has no is_right)."""
    p=fr.get('pose') if isinstance(fr,dict) else None
    if not p or p.get('hands') is None: return [], [], frozenset()
    raw=np.asarray(p['hands']); ir=p.get('hands_is_right')
    ir=np.asarray(ir) if ir is not None else None
    rw = _wrist(p,4) if ir is None else None    # R wrist
    lw = _wrist(p,7) if ir is None else None    # L wrist
    hands=[]; labs=[]
    for i,x in enumerate(raw):
        h=vh(x)
        if h is None: continue
        if ir is not None and i<len(ir):
            lab=bool(ir[i])
        elif rw is not None and lw is not None:
            lab = np.linalg.norm(h[0]-rw) < np.linalg.norm(h[0]-lw)   # True=Right (closer to R wrist)
        else:
            lab=None
        hands.append(h); labs.append(lab)
    # presence key: set of labels if all labelled, else the count
    if hands and all(l is not None for l in labs):
        key=frozenset(labs) if len(set(labs))==len(labs) else ("n",len(hands))  # dup labels -> use count
    else:
        key=("n",len(hands))
    return hands, labs, key

def jump(cur,curL,prev,prevL):
    if not cur or not prev: return np.nan
    vals=[]
    # try label match first
    if all(l is not None for l in curL) and all(l is not None for l in prevL) and len(set(curL))==len(curL) and len(set(prevL))==len(prevL):
        pm={l:h for h,l in zip(prev,prevL)}
        for h,l in zip(cur,curL):
            if l in pm:
                b=pm[l]; sc=(hsz(h)+hsz(b))/2
                vals.append(float((np.linalg.norm(h-b,axis=1)/sc).mean()))
    else:  # nearest-root fallback
        used=set()
        for h in cur:
            best=None;bd=1e9
            for i,b in enumerate(prev):
                if i in used: continue
                dd=np.linalg.norm(h[0]-b[0])
                if dd<bd:bd=dd;best=i
            if best is not None:
                used.add(best); b=prev[best]; sc=(hsz(h)+hsz(b))/2
                vals.append(float((np.linalg.norm(h-b,axis=1)/sc).mean()))
    return float(np.median(vals)) if vals else np.nan

def runs(mask, minrun):
    out=[]; s=None
    for i,v in enumerate(list(mask)+[False]):
        if v and s is None: s=i
        elif not v and s is not None:
            if (i-s)>=minrun: out.append([s,i-1,round(s/FPS,2),round(i/FPS,2)])
            s=None
    return out

def process(cid, src):
    d=pickle.load(open(f"{SRCDIR[src]}/{cid}_dwpose.pkl",'rb'))
    n=len(d); jvals=[]; jmask=[False]*n; pmask=[False]*n
    prev=prevL=None; prevkey=None
    for k in range(n):
        cur,curL,key=get_hands(d[k])
        if k>0:
            jv=jump(cur,curL,prev,prevL)
            if np.isfinite(jv): jvals.append(jv); jmask[k]=jv>THRESH
            if prevkey is not None and key!=prevkey: pmask[k]=True   # presence change
        prev,prevL,prevkey=cur,curL,key
    bad=[jmask[i] or pmask[i] for i in range(n)]
    def pct(q): return round(float(np.percentile(jvals,q)),4) if jvals else None
    rec=dict(clip=cid, n_frames=n, source=src,
             jump_p50=pct(50), jump_p95=pct(95), jump_max=round(float(max(jvals)),4) if jvals else None,
             n_jump_bad=int(sum(jmask)), jump_segments=runs(jmask,MIN_RUN),
             n_presence_change=int(sum(pmask)), presence_frames=[i for i in range(n) if pmask[i]],
             n_bad=int(sum(bad)), bad_segments=runs(bad,1))
    json.dump(rec, open(f"{OUT[src]}/{cid}.json","w"))
    return rec

def all_ids(src):
    D=SRCDIR[src]
    ids=set(f[:-len('_dwpose.pkl')] for f in os.listdir(D) if f.endswith('_dwpose.pkl'))
    if src=="dwpose":  # match the 95k packed set
        ids &= set(f[:-len('_dwpose.pkl')] for f in os.listdir(VC) if f.endswith('_dwpose.pkl'))
    return sorted(ids)

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument('--source',required=True)
    ap.add_argument('--shard',type=int,default=0); ap.add_argument('--nshards',type=int,default=1)
    a=ap.parse_args(); os.makedirs(OUT[a.source],exist_ok=True)
    ids=all_ids(a.source); mine=[c for i,c in enumerate(ids) if i%a.nshards==a.shard]
    print(f"{a.source} shard {a.shard}/{a.nshards}: {len(mine)} clips",flush=True)
    done=0
    for cid in mine:
        try: process(cid,a.source)
        except Exception: pass
        done+=1
        if done%2000==0: print(f"  {a.source} shard{a.shard}: {done}/{len(mine)}",flush=True)
    print(f"{a.source}_SHARD_{a.shard}_DONE",flush=True)
