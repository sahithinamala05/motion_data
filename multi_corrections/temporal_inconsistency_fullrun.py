"""Full-run temporal-inconsistency check for WiLoR hands, DWpose hands, DWpose body.

For every clip (WiLoR from video_cut_wilor / video_cut_wilor_rem, DWpose from
live_dealer_blackjack/dwpose/batch_01/dwpose), for each frame that is NOT blurry
(Laplacian variance >= BLUR_LAP), measure frame-to-frame displacement of keypoints vs
the previous NON-blur frame, scale-normalized:
  temp_wilor  : median over WiLoR hands of mean_kpt||p_t - p_{t-1}|| / handsize
  temp_dwpose : same for DWpose hands
  temp_body   : median over wrists+elbows of ||p_t - p_{t-1}|| / shoulder_width
A frame is temporally-inconsistent for a source if its value > THRESH (hands 0.5,
body 0.3). Consecutive bad frames -> segments (isolated 1-frame drops, MIN_RUN=2).

Per clip -> OUT/<cid>.json:
  {clip, n_frames, n_blur, temp_wilor_p95, temp_dwpose_p95, temp_body_p95,
   wilor_segments, dwpose_segments, body_segments}   (segments: [start_f,end_f,start_s,end_s])
Also appends a summary row per clip to OUT_SUMMARY (parquet, written by launcher shards
then merged). Sharded via --shard i --nshards N for parallelism. CPU-only.
"""
import os, sys, json, pickle, argparse
import numpy as np, cv2; cv2.setNumThreads(1)

VC_DIRS=["/home/ubuntu/us-west-3-fs/sahithi/Wilor/video_cut_wilor",
         "/home/ubuntu/us-west-3-fs/sahithi/Wilor/video_cut_wilor_rem"]
DW="/home/ubuntu/us-west-3-fs/live_dealer_blackjack/dwpose/batch_01/dwpose"
VID="/home/ubuntu/us-west-3-fs/live_dealer_blackjack/video_cut/batch_01"
OUT="/home/ubuntu/us-west-3-fs/sahithi/hand_consistency/temporal"
BLUR_LAP=150; FPS=30.0
TH_HAND=0.5; TH_BODY=0.3; MIN_RUN=2
RWRI,LWRI,RELB,LELB,RSHO,LSHO=4,7,3,6,2,5

def wilor_path(cid):
    for d in VC_DIRS:
        p=f"{d}/{cid}_dwpose.pkl"
        if os.path.exists(p): return p
    return None
def valid_hand(h):
    h=np.asarray(h,float)
    if h.ndim!=2 or h.shape!=(21,2): return None
    return h if (np.isfinite(h).all() and (np.ptp(h[:,0])+np.ptp(h[:,1]))>1e-4) else None
def handsize(h): return max(np.ptp(h[:,0]),np.ptp(h[:,1])) or 1e-3
def body_pt(c,s,j):
    i=int(s[0,j])
    if i<0 or i>=len(c): return None
    x,y=c[i]; return np.array([x,y]) if np.isfinite(x) and np.isfinite(y) else None
def match(A,B):
    pairs=[]; used=set()
    for a in A:
        best=None;bd=1e9
        for ib,b in enumerate(B):
            if ib in used: continue
            d=np.linalg.norm(a[0]-b[0])
            if d<bd: bd=d;best=ib
        if best is not None: used.add(best); pairs.append((a,B[best]))
    return pairs
def temporal(cur,prev):
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

def lap_of(cid,n):
    p=f"{VID}/{cid}.mp4"
    if not os.path.exists(p): return None
    cap=cv2.VideoCapture(p); out=[]
    while len(out)<n:
        ok,img=cap.read()
        if not ok: break
        out.append(float(cv2.Laplacian(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var()))
    cap.release()
    while len(out)<n: out.append(np.nan)
    return np.array(out)

def process(cid):
    pw=wilor_path(cid); pd=f"{DW}/{cid}_dwpose.pkl"
    if not (pw and os.path.exists(pd)): return None
    try: W=pickle.load(open(pw,'rb')); D=pickle.load(open(pd,'rb'))
    except Exception: return None
    n=min(len(W),len(D)); lap=lap_of(cid,n)
    tw=[];td=[];tb=[]; mw=[False]*n; md=[False]*n; mb=[False]*n; n_blur=0
    prevW=None;prevD=None;prevB={}
    for k in range(n):
        if lap is not None and np.isfinite(lap[k]) and lap[k]<BLUR_LAP:
            n_blur+=1; prevW=None;prevD=None;prevB={}; continue
        pWa=W[k]['pose']; pDa=D[k]['pose']
        Wh=[h for h in [valid_hand(x) for x in np.asarray(pWa.get('hands'))] if h is not None] if pWa.get('hands') is not None else []
        Dh=[h for h in [valid_hand(x) for x in np.asarray(pDa.get('hands'))] if h is not None] if pDa.get('hands') is not None else []
        twf=temporal(Wh,prevW); tdf=temporal(Dh,prevD)
        c=np.asarray(pDa['bodies']['candidate']); s=np.asarray(pDa['bodies']['subset'])
        rs,ls=body_pt(c,s,RSHO),body_pt(c,s,LSHO)
        sw=np.linalg.norm(rs-ls) if (rs is not None and ls is not None) else np.nan
        curB={j:body_pt(c,s,j) for j in (RWRI,LWRI,RELB,LELB)}; curB={j:p for j,p in curB.items() if p is not None}
        bd=[np.linalg.norm(curB[j]-prevB[j])/sw for j in curB if j in prevB] if (np.isfinite(sw) and sw>0) else []
        tbf=float(np.median(bd)) if bd else np.nan
        if np.isfinite(twf): tw.append(twf); mw[k]=twf>TH_HAND
        if np.isfinite(tdf): td.append(tdf); md[k]=tdf>TH_HAND
        if np.isfinite(tbf): tb.append(tbf); mb[k]=tbf>TH_BODY
        prevW=Wh;prevD=Dh;prevB=curB
    def p95(a): return round(float(np.percentile(a,95)),4) if a else None
    rec=dict(clip=cid, n_frames=n, n_blur=n_blur,
             temp_wilor_p95=p95(tw), temp_dwpose_p95=p95(td), temp_body_p95=p95(tb),
             wilor_segments=segs(mw), dwpose_segments=segs(md), body_segments=segs(mb))
    json.dump(rec, open(f"{OUT}/{cid}.json","w"))
    return rec

def all_ids():
    s=set()
    for d in VC_DIRS:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.endswith('_dwpose.pkl'): s.add(f[:-len('_dwpose.pkl')])
    return sorted(s)

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument('--shard',type=int,default=0); ap.add_argument('--nshards',type=int,default=1)
    a=ap.parse_args()
    os.makedirs(OUT,exist_ok=True)
    ids=all_ids(); mine=[c for i,c in enumerate(ids) if i%a.nshards==a.shard]
    print(f"shard {a.shard}/{a.nshards}: {len(mine)} clips (of {len(ids)})",flush=True)
    done=0
    for cid in mine:
        if os.path.exists(f"{OUT}/{cid}.json"): done+=1; continue  # resume
        try: process(cid)
        except Exception as e: pass
        done+=1
        if done%1000==0: print(f"  shard{a.shard}: {done}/{len(mine)}",flush=True)
    print(f"SHARD_{a.shard}_DONE",flush=True)
