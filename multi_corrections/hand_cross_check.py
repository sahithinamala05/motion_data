"""Hand-cross detection on WiLoR+DWpose pkls.

For each frame that has >=2 hands, take the hand pair with the most overlap and measure:
  overlap_pts = #keypoints of A inside B's bbox + #keypoints of B inside A's bbox  (0..42)
  iou         = IoU of the two hand bounding boxes
A frame is a HAND CROSS if  iou >= IOU_T  AND  overlap_pts >= PTS_T
(adjacency alone -> iou ~0 -> not counted; a real cross has the hands' boxes overlapping
and many points landing inside the other hand).

Consecutive cross frames are collapsed into segments. One JSON per video is written
ONLY for videos that have >=1 hand-cross segment, to OUT/<cid>.json:
  { base_name, source, n_frames, frame_dimensions, fps,
    iou_thresh, pts_thresh, n_cross_frames,
    hand_cross_segments: [[start_f, end_f, start_s, end_s], ...],
    per_segment: [{start_f,end_f,peak_pts,peak_iou}, ...] }

Processes BOTH:
  /home/ubuntu/us-west-3-fs/sahithi/Wilor/video_cut_wilor        (*_dwpose.pkl)
  /home/ubuntu/us-west-3-fs/sahithi/Wilor/video_cut_wilor_rem_new (*_dealer.pkl)
De-dups on clip id (a clip present in both is processed once, preferring rem_new).
"""
import os, sys, json, pickle
import numpy as np
from multiprocessing import Pool

SRC=[("/home/ubuntu/us-west-3-fs/sahithi/Wilor/video_cut_wilor_rem_new","_dealer.pkl"),
     ("/home/ubuntu/us-west-3-fs/sahithi/Wilor/video_cut_wilor","_dwpose.pkl")]
OUT="/home/ubuntu/us-west-3-fs/sahithi/hand_cross_check"
FPS=30.0
IOU_T=0.50   # hand boxes must overlap >=50%
PTS_T=10
MIN_RUN=5    # require sustained overlap: a segment must be >= MIN_RUN consecutive frames
             # (drops brief touches; run-length dist is bimodal, real crosses hold 30+ frames)

def hand_valid(h):
    ok=np.isfinite(h).all(1)&(h[:,0]>-0.05)&(h[:,0]<1.05)&(h[:,1]>-0.05)&(h[:,1]<1.05)
    return h[ok] if ok.sum()>=5 else None
def bbox(h): return (h[:,0].min(),h[:,1].min(),h[:,0].max(),h[:,1].max())
def iou(a,b):
    ix0,iy0=max(a[0],b[0]),max(a[1],b[1]); ix1,iy1=min(a[2],b[2]),min(a[3],b[3])
    inter=max(0.0,ix1-ix0)*max(0.0,iy1-iy0)
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0.0
def inside(pts,bb): return int(((pts[:,0]>=bb[0])&(pts[:,0]<=bb[2])&(pts[:,1]>=bb[1])&(pts[:,1]<=bb[3])).sum())

def frame_metrics(pose):
    hh=pose.get('hands') if pose else None
    if hh is None: return 0,0.0
    hh=np.asarray(hh)
    if hh.ndim!=3 or hh.shape[0]<2: return 0,0.0
    hands=[hand_valid(hh[k]) for k in range(hh.shape[0])]
    bp=0; bi=0.0
    for i in range(len(hands)):
        for j in range(i+1,len(hands)):
            A,B=hands[i],hands[j]
            if A is None or B is None: continue
            bbA,bbB=bbox(A),bbox(B)
            pts=inside(A,bbB)+inside(B,bbA); ii=iou(bbA,bbB)
            if pts>bp: bp=pts
            if ii>bi: bi=ii
    return bp,bi

def segments(mask,P,I):
    out=[]; s=None
    for i,v in enumerate(list(mask)+[False]):
        if v and s is None: s=i
        elif not v and s is not None:
            if (i-s)>=MIN_RUN:   # drop isolated 1-frame crosses
                out.append(dict(start_f=s,end_f=i-1,start_s=round(s/FPS,2),end_s=round(i/FPS,2),
                                peak_pts=int(P[s:i].max()),peak_iou=round(float(I[s:i].max()),3)))
            s=None
    return out

def analyze(arg):
    path,cid,src=arg
    try: d=pickle.load(open(path,'rb'))
    except Exception: return None
    P=np.zeros(len(d),int); I=np.zeros(len(d),float)
    for k,fr in enumerate(d):
        p=fr.get('pose') if isinstance(fr,dict) else None
        bp,bi=frame_metrics(p); P[k]=bp; I[k]=bi
    mask=(I>=IOU_T)&(P>=PTS_T)
    if not mask.any(): return (cid,0)   # no cross -> not saved
    segs=segments(mask,P,I)             # 1-frame crosses already dropped
    if not segs: return (cid,0)         # only isolated single-frame crosses -> not saved
    n_cross=sum(s['end_f']-s['start_f']+1 for s in segs)
    dims=d[0].get('frame_dimensions') if isinstance(d[0],dict) else None
    rec=dict(base_name=cid+".mp4", source=src, n_frames=len(d),
             frame_dimensions=list(dims) if dims is not None else None, fps=FPS,
             iou_thresh=IOU_T, pts_thresh=PTS_T, min_run=MIN_RUN, n_cross_frames=int(n_cross),
             hand_cross_segments=[[s['start_f'],s['end_f'],s['start_s'],s['end_s']] for s in segs],
             per_segment=segs)
    with open(f"{OUT}/{cid}.json","w") as f: json.dump(rec,f)
    return (cid,int(mask.sum()))

def build_worklist():
    seen=set(); work=[]
    for base,suf in SRC:
        if not os.path.isdir(base): continue
        for f in os.listdir(base):
            if not f.endswith(suf): continue
            cid=f[:-len(suf)]
            if cid in seen: continue
            seen.add(cid); work.append((f"{base}/{f}",cid,os.path.basename(base)))
    return work

if __name__=="__main__":
    os.makedirs(OUT,exist_ok=True)
    if len(sys.argv)>1 and sys.argv[1]=="--viz":
        import viz_hand_cross  # separate helper
        sys.exit(0)
    work=build_worklist()
    print(f"clips to scan (de-duped): {len(work)}  rule: iou>={IOU_T} AND pts>={PTS_T}",flush=True)
    saved=0; withcross=0; done=0
    with Pool(24) as pool:
        for r in pool.imap_unordered(analyze,work,chunksize=32):
            done+=1
            if r and r[1]>0: withcross+=1; saved+=1
            if done%10000==0: print(f"  {done}/{len(work)}  saved(with cross)={saved}",flush=True)
    print(f"DONE scanned {done}; JSONs written (videos WITH cross) = {saved} -> {OUT}",flush=True)
