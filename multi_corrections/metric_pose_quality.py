"""Prototype pose-quality metrics on the isolated single-dealer pkls (full_run/pkl).
No confidence available -> geometry + temporal + presence. Per clip:
  presence : frac_body, frac_2hands, med_valid_body(0-18), med_face_valid(0-68)
  attach   : hand-root -> nearest DWpose wrist distance / forearm  (floating/mislinked hands)
  bonecov  : temporal coeff-of-variation of forearm/shoulder bone lengths (instability/jitter)
  jitter   : 95th-pct frame-to-frame wrist/elbow displacement / shoulder-width (teleporting)
Composite 'badness' = sum of z-scores of the bad-direction metrics. Ranks worst + vizzes them.
"""
import os, sys, glob, pickle, random, subprocess
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, "/home/ubuntu/us-west-3-fs/sahithi/_entire_code/multi_corrections")
from apply_quality_to_pkl import draw_skeleton
import cv2; cv2.setNumThreads(1)

PKL = "/home/ubuntu/us-west-3-fs/sahithi/multi_correction/full_run/pkl"
VC  = "/home/ubuntu/us-west-3-fs/live_dealer_blackjack/video_cut/batch_01"
OUT = "/home/ubuntu/us-west-3-fs/sahithi/rem_95k_filtered/pose_quality"
VIS = os.path.join(OUT, "vis_worst")
# OpenPose-18 indices
RSHO,RELB,RWRI,LSHO,LELB,LWRI = 2,3,4,5,6,7
FPS=30.0

def clip_metrics(cid):
    try:
        d = pickle.load(open(f"{PKL}/{cid}_dealer.pkl", 'rb'))
    except Exception:
        return None
    H, W = d[0]['frame_dimensions']
    nfr = len(d); body=0; hand2=0; face=0
    nvalid=[]; faceval=[]; att=[]
    bones={'faR':[],'faL':[],'sh':[]}; jit=[]; prev={}
    def gp(cand, sub, j):
        i=int(sub[0,j])
        if i<0 or i>=len(cand): return None
        x,y=cand[i]
        return np.array([x*W, y*H]) if (np.isfinite(x) and np.isfinite(y)) else None
    def dist(a,b): return float(np.linalg.norm(a-b)) if (a is not None and b is not None) else np.nan
    for fr in d:
        pose=fr.get('pose')
        if pose is None or pose.get('bodies') is None: continue
        cand=np.asarray(pose['bodies']['candidate']); sub=np.asarray(pose['bodies']['subset'])
        if sub.ndim!=2: continue
        body+=1; nvalid.append(int((sub[0]>=0).sum()))
        f=pose.get('faces')
        if f is not None:
            f=np.asarray(f)
            if f.size:
                v=np.isfinite(f[0]).all(1)&(f[0,:,0]>=0)&(f[0,:,0]<=1)&(f[0,:,1]>=0)&(f[0,:,1]<=1)
                faceval.append(int(v.sum())); face += v.sum()>=60
        rs,re,rw=gp(cand,sub,RSHO),gp(cand,sub,RELB),gp(cand,sub,RWRI)
        ls,le,lw=gp(cand,sub,LSHO),gp(cand,sub,LELB),gp(cand,sub,LWRI)
        faR,faL,sh=dist(re,rw),dist(le,lw),dist(rs,ls)
        bones['faR'].append(faR); bones['faL'].append(faL); bones['sh'].append(sh)
        scale = sh if np.isfinite(sh) and sh>0 else np.nanmean([faR,faL])
        hh=pose.get('hands')
        if hh is not None:
            hh=np.asarray(hh)
            if hh.ndim==3 and hh.shape[0]>0:
                hand2 += hh.shape[0]>=2
                wr=[w for w in (rw,lw) if w is not None]; fa=np.nanmean([faR,faL])
                if wr and np.isfinite(fa) and fa>0:
                    for k in range(hh.shape[0]):
                        root=hh[k,0]*[W,H]
                        att.append(min(np.linalg.norm(root-w) for w in wr)/fa)
        cur={}
        for nm,j in (('rw',RWRI),('lw',LWRI),('re',RELB),('le',LELB)):
            p=gp(cand,sub,j)
            if p is not None: cur[nm]=p
        if np.isfinite(scale) and scale>0:
            for nm,p in cur.items():
                if nm in prev: jit.append(np.linalg.norm(p-prev[nm])/scale)
        prev=cur
    def cov(x):
        x=np.array([v for v in x if np.isfinite(v)])
        return float(np.std(x)/np.median(x)) if len(x)>=5 and np.median(x)>0 else np.nan
    return dict(cid=cid, n_frames=nfr, frac_body=body/max(nfr,1), frac_2hands=hand2/max(body,1),
        med_valid_body=float(np.median(nvalid)) if nvalid else 0.0,
        med_face_valid=float(np.median(faceval)) if faceval else 0.0,
        att_p90=float(np.nanpercentile(att,90)) if att else np.nan,
        bonecov=float(np.nanmean([cov(bones['faR']),cov(bones['faL']),cov(bones['sh'])])),
        jitter_p95=float(np.nanpercentile(jit,95)) if jit else np.nan)

def render_worst(row):
    cid=row['cid']; d=pickle.load(open(f"{PKL}/{cid}_dealer.pkl",'rb'))
    cap=cv2.VideoCapture(f"{VC}/{cid}.mp4"); w=int(cap.get(3)) or 1920; h=int(cap.get(4)) or 1080
    ff=subprocess.Popen(["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pix_fmt","bgr24","-s",f"{w}x{h}",
        "-r","30","-i","pipe:0","-c:v","libx264","-pix_fmt","yuv420p","-crf","22","-movflags","+faststart",
        f"{VIS}/{cid}_mid.mp4"],stdin=subprocess.PIPE)
    idx=0
    while True:
        ok,img=cap.read()
        if not ok: break
        if (img.shape[1],img.shape[0])!=(w,h): img=cv2.resize(img,(w,h))
        draw_skeleton(img, d[idx].get('pose') if idx<len(d) else None, w,h)
        cv2.rectangle(img,(0,0),(w-1,56),(0,0,0),-1)
        cv2.putText(img,f"att_p90 {row['att_p90']:.1f}  bonecov {row['bonecov']:.2f}  jit95 {row['jitter_p95']:.2f}  body {row['frac_body']:.2f}  bad {row['bad']:.1f}",
                    (12,38),cv2.FONT_HERSHEY_SIMPLEX,0.85,(0,0,255),2)
        ff.stdin.write(np.ascontiguousarray(img).tobytes()); idx+=1
    cap.release(); ff.stdin.close(); ff.wait(); return cid

if __name__=="__main__":
    os.makedirs(VIS,exist_ok=True)
    names=[f[:-len('_dealer.pkl')] for f in os.listdir(PKL) if f.endswith('_dealer.pkl')]
    by_date={}
    for c in names: by_date.setdefault(c.split('_')[0],[]).append(c)
    for d in by_date: random.Random(1).shuffle(by_date[d])
    dates=sorted(by_date); samp=[]; i=0
    while len(samp)<2000 and any(by_date.values()):
        dd=dates[i%len(dates)]
        if by_date[dd]: samp.append(by_date[dd].pop(0))
        i+=1
    print(f"computing metrics on {len(samp)} clips ...",flush=True)
    with Pool(16) as pool:
        rows=[r for r in pool.map(clip_metrics,samp,chunksize=8) if r]
    df=pd.DataFrame(rows)
    # composite badness = sum of z-scores of bad-direction metrics
    def z(col, hi_bad=True):
        x=df[col].astype(float); m,s=x.mean(),x.std()
        zz=(x-m)/s if s>0 else x*0
        return zz if hi_bad else -zz
    df['bad']= (z('att_p90')+z('bonecov')+z('jitter_p95')
                + z('frac_body',hi_bad=False) + z('med_valid_body',hi_bad=False)).round(2)
    df=df.sort_values('bad',ascending=False)
    os.makedirs(OUT,exist_ok=True); df.to_csv(f"{OUT}/pose_quality_sample.csv",index=False)
    print("\n=== metric summary (sample) ===")
    print(df[['frac_body','frac_2hands','med_valid_body','att_p90','bonecov','jitter_p95']].describe().round(2).to_string())
    print("\n=== 12 WORST (highest badness) ===")
    print(df.head(12)[['cid','frac_body','att_p90','bonecov','jitter_p95','bad']].to_string(index=False))
    worst=df.head(12)
    print(f"\nrendering {len(worst)} worst -> {VIS}",flush=True)
    for _,r in worst.iterrows():
        try: render_worst(r); print("  viz",r['cid'],flush=True)
        except Exception as e: print("  viz ERR",r['cid'],e,flush=True)
    print("DONE",flush=True)
