"""PER-FRAME pose-quality check on video_cut_wilor_rem_new (quality-cleaned dealer pkls).
For each frame that HAS a pose, decide good/bad-pose using absolute thresholds:
  float_hand  : hand root > ATT_T forearms from nearest DWpose wrist
  jitter      : a tracked joint jumped > JIT_T shoulder-widths from the previous frame
  bone_outlier: a bone length deviates > BONE_DEV from the clip's own median bone length
  sparse      : < MIN_BODY valid body joints
Bad-pose frames are collapsed into problematic_pose_segments [start_f,end_f,start_s,end_s]
(same format as the dark/blur problematic_segments), + per-clip reason counts. Frames with
pose=None are already removed (dark/blur/not-found) -> not re-assessed here.
Two passes per clip: (1) collect bone lengths -> clip medians, (2) per-frame flags.
"""
import os, sys, glob, pickle, random, subprocess
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, "/home/ubuntu/us-west-3-fs/sahithi/_entire_code/multi_corrections")
from apply_quality_to_pkl import draw_skeleton
import cv2; cv2.setNumThreads(1)

PKL = "/home/ubuntu/us-west-3-fs/sahithi/Wilor/video_cut_wilor_rem_new"
VC  = "/home/ubuntu/us-west-3-fs/live_dealer_blackjack/video_cut/batch_01"
OUT = "/home/ubuntu/us-west-3-fs/sahithi/rem_95k_filtered/pose_quality_perframe"
VIS = os.path.join(OUT, "vis_worst")
RSHO,RELB,RWRI,LSHO,LELB,LWRI = 2,3,4,5,6,7
FPS = 30.0
# absolute per-frame thresholds (first-pass defaults; tune against viz)
ATT_T, JIT_T, BONE_DEV, MIN_BODY = 2.0, 0.5, 0.5, 8

def _px(cand, sub, j, W, H):
    i=int(sub[0,j])
    if i<0 or i>=len(cand): return None
    x,y=cand[i]
    return np.array([x*W,y*H]) if (np.isfinite(x) and np.isfinite(y)) else None
def _d(a,b): return float(np.linalg.norm(a-b)) if (a is not None and b is not None) else np.nan

def per_frame_flags(d):
    """returns list of (has_pose, bad, reasons) per frame + reason counters"""
    H,W = d[0]['frame_dimensions']
    # ---- pass 1: per-frame raw values ----
    faR=[];faL=[];sho=[]
    for fr in d:
        p=fr.get('pose')
        if p is None or p.get('bodies') is None: faR.append(np.nan);faL.append(np.nan);sho.append(np.nan); continue
        c=np.asarray(p['bodies']['candidate']); s=np.asarray(p['bodies']['subset'])
        re,rw,le,lw,rs,ls=_px(c,s,RELB,W,H),_px(c,s,RWRI,W,H),_px(c,s,LELB,W,H),_px(c,s,LWRI,W,H),_px(c,s,RSHO,W,H),_px(c,s,LSHO,W,H)
        faR.append(_d(re,rw)); faL.append(_d(le,lw)); sho.append(_d(rs,ls))
    med={'faR':np.nanmedian(faR) if np.any(np.isfinite(faR)) else np.nan,
         'faL':np.nanmedian(faL) if np.any(np.isfinite(faL)) else np.nan,
         'sho':np.nanmedian(sho) if np.any(np.isfinite(sho)) else np.nan}
    # ---- pass 2: per-frame decision ----
    flags=[]; prev={}; cnt=dict(float_hand=0,jitter=0,bone_outlier=0,sparse=0)
    for idx,fr in enumerate(d):
        p=fr.get('pose')
        if p is None or p.get('bodies') is None:
            flags.append((False,False,[])); prev={}; continue
        c=np.asarray(p['bodies']['candidate']); s=np.asarray(p['bodies']['subset'])
        reasons=[]
        # sparse
        if int((s[0]>=0).sum()) < MIN_BODY: reasons.append('sparse')
        rw,lw=_px(c,s,RWRI,W,H),_px(c,s,LWRI,W,H); re,le=_px(c,s,RELB,W,H),_px(c,s,LELB,W,H)
        fa=np.nanmean([faR[idx],faL[idx]])
        # float hand
        hh=p.get('hands')
        if hh is not None:
            hh=np.asarray(hh); wr=[w for w in (rw,lw) if w is not None]
            if hh.ndim==3 and hh.shape[0]>0 and wr and np.isfinite(fa) and fa>0:
                worst=max(min(np.linalg.norm(hh[k,0]*[W,H]-w) for w in wr) for k in range(hh.shape[0]))/fa
                if worst>ATT_T: reasons.append('float_hand')
        # NOTE: per-frame bone-length "outlier" removed — 2D foreshortening makes bone length
        # legitimately vary a lot frame-to-frame (false positives). Bone instability is kept
        # only as the CLIP-level bonecov signal, not a per-frame flag.
        # jitter vs previous frame
        scale = sho[idx] if np.isfinite(sho[idx]) and sho[idx]>0 else fa
        cur={}
        for nm,j in (('rw',RWRI),('lw',LWRI),('re',RELB),('le',LELB)):
            q=_px(c,s,j,W,H)
            if q is not None: cur[nm]=q
        if np.isfinite(scale) and scale>0:
            for nm,q in cur.items():
                if nm in prev and np.linalg.norm(q-prev[nm])/scale > JIT_T: reasons.append('jitter'); break
        prev=cur
        bad=len(reasons)>0
        for r in set(reasons): cnt[r]+=1
        flags.append((True,bad,reasons))
    return flags, cnt

def intervals(bad):
    out=[]; s=None
    for i,v in enumerate(bad):
        if v and s is None: s=i
        elif not v and s is not None: out.append([s,i-1,round(s/FPS,2),round(i/FPS,2)]); s=None
    if s is not None: out.append([s,len(bad)-1,round(s/FPS,2),round(len(bad)/FPS,2)])
    return out

def clip(cid):
    try: d=pickle.load(open(f"{PKL}/{cid}_dealer.pkl",'rb'))
    except Exception: return None
    flags,cnt=per_frame_flags(d)
    npose=sum(1 for h,_,_ in flags if h); nbad=sum(1 for _,b,_ in flags if b)
    bad=[b for _,b,_ in flags]
    return dict(base_name=cid+".mp4", n_frames=len(d), n_pose_frames=npose, n_bad_pose=nbad,
                frac_bad_pose=round(nbad/npose,4) if npose else 0.0,
                problematic_pose_segments=intervals(bad),
                n_float_hand=cnt['float_hand'], n_jitter=cnt['jitter'],
                n_bone_outlier=cnt['bone_outlier'], n_sparse=cnt['sparse'])

def render(cid):
    d=pickle.load(open(f"{PKL}/{cid}_dealer.pkl",'rb')); flags,_=per_frame_flags(d)
    cap=cv2.VideoCapture(f"{VC}/{cid}.mp4"); w=int(cap.get(3)) or 1920; h=int(cap.get(4)) or 1080
    ff=subprocess.Popen(["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pix_fmt","bgr24","-s",f"{w}x{h}",
        "-r","30","-i","pipe:0","-c:v","libx264","-pix_fmt","yuv420p","-crf","22","-movflags","+faststart",
        f"{VIS}/{cid}_mid.mp4"],stdin=subprocess.PIPE)
    idx=0
    while True:
        ok,img=cap.read()
        if not ok: break
        if (img.shape[1],img.shape[0])!=(w,h): img=cv2.resize(img,(w,h))
        has,bad,reasons = flags[idx] if idx<len(flags) else (False,False,[])
        draw_skeleton(img, d[idx].get('pose') if idx<len(d) else None, w,h)
        if not has: col,txt=(150,150,150),"no pose (dark/blur)"
        elif bad:  col,txt=(0,0,255),"BAD POSE: "+",".join(sorted(set(reasons)))
        else:      col,txt=(0,255,0),"POSE OK"
        cv2.rectangle(img,(0,0),(w-1,52),(0,0,0),-1)
        cv2.putText(img,f"f{idx} {idx/FPS:4.1f}s  {txt}",(12,38),cv2.FONT_HERSHEY_SIMPLEX,0.95,col,2)
        ff.stdin.write(np.ascontiguousarray(img).tobytes()); idx+=1
    cap.release(); ff.stdin.close(); ff.wait(); return cid

if __name__=="__main__":
    os.makedirs(VIS,exist_ok=True)
    names=[f[:-len('_dealer.pkl')] for f in os.listdir(PKL) if f.endswith('_dealer.pkl')]
    by_date={}
    for c in names: by_date.setdefault(c.split('_')[0],[]).append(c)
    for dd in by_date: random.Random(2).shuffle(by_date[dd])
    dates=sorted(by_date); samp=[]; i=0
    while len(samp)<2000 and any(by_date.values()):
        d0=dates[i%len(dates)]
        if by_date[d0]: samp.append(by_date[d0].pop(0))
        i+=1
    print(f"per-frame pose check on {len(samp)} clips (thresholds att>{ATT_T} jit>{JIT_T} bonedev>{BONE_DEV} minbody<{MIN_BODY})",flush=True)
    with Pool(16) as pool:
        rows=[r for r in pool.map(clip,samp,chunksize=8) if r]
    df=pd.DataFrame(rows); os.makedirs(OUT,exist_ok=True)
    df.to_parquet(f"{OUT}/pose_quality_perframe_sample.parquet")
    print("\n=== per-frame flag summary (sample) ===")
    print(f"clips: {len(df)}")
    print(f"clips with >=1 bad-pose frame: {(df['n_bad_pose']>0).sum()} ({100*(df['n_bad_pose']>0).mean():.1f}%)")
    tot_pose=df['n_pose_frames'].sum(); tot_bad=df['n_bad_pose'].sum()
    print(f"pose frames: {tot_pose:,}  bad-pose frames: {tot_bad:,} ({100*tot_bad/max(tot_pose,1):.2f}%)")
    print("reason totals:", {k:int(df['n_'+k].sum()) for k in ('float_hand','jitter','bone_outlier','sparse')})
    print("\n=== 10 worst by frac_bad_pose (>=100 pose frames) ===")
    w=df[df['n_pose_frames']>=100].sort_values('frac_bad_pose',ascending=False).head(10)
    print(w[['base_name','n_pose_frames','n_bad_pose','frac_bad_pose','n_float_hand','n_jitter','n_bone_outlier']].to_string(index=False))
    print(f"\nrendering {len(w)} worst -> {VIS}",flush=True)
    for cid in [b[:-4] for b in w['base_name']]:
        try: render(cid); print("  viz",cid,flush=True)
        except Exception as e: print("  ERR",cid,e,flush=True)
    print("DONE",flush=True)
