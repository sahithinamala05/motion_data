"""Visualize the worst (jumpiest) clips per temporal source into hand_consistency/vis/<src>/.
src in {dwpose, wilor, body}. Draws the source's keypoints; frames inside a bad
temporal segment are flagged red with the per-frame jump value.
"""
import os, sys, json, pickle, subprocess, argparse
import numpy as np, cv2; cv2.setNumThreads(1)
from multiprocessing import Pool

BASE="/home/ubuntu/us-west-3-fs/sahithi/hand_consistency"
VC="/home/ubuntu/us-west-3-fs/sahithi/Wilor/video_cut_wilor"
DW="/home/ubuntu/us-west-3-fs/live_dealer_blackjack/dwpose/batch_01/dwpose"
VID="/home/ubuntu/us-west-3-fs/live_dealer_blackjack/video_cut/batch_01"
FPS=30.0
RWRI,LWRI,RELB,LELB,RSHO,LSHO=4,7,3,6,2,5
# OpenPose-18 skeleton, NOT below the hip (no knees/ankles)
BODY_EDGES=[(1,2),(1,5),(2,3),(3,4),(5,6),(6,7),(1,8),(1,11),
            (1,0),(0,14),(14,16),(0,15),(15,17)]
BODY_ALLOWED={0,1,2,3,4,5,6,7,8,11,14,15,16,17}
CFG={  # src -> (json_dir, pkl_dir, kind, p95key, thresh)
 "dwpose":("dwpose_temporal", DW, "hand", "jump_p95", 1.0),
 "wilor" :("wilor_temporal",  VC, "hand", "jump_p95", 1.0),
 "body"  :("body_temporal",   DW, "body", "p95",      1.0),
}

def valid_hand(h):
    h=np.asarray(h,float)
    if h.ndim!=2 or h.shape!=(21,2): return None
    return h if (np.isfinite(h).all() and (np.ptp(h[:,0])+np.ptp(h[:,1]))>1e-4) else None
def handsize(h): return max(np.ptp(h[:,0]),np.ptp(h[:,1])) or 1e-3
def match(cur,prev):
    pairs=[];used=set()
    for a in cur:
        best=None;bd=1e9
        for i,b in enumerate(prev):
            if i in used: continue
            d=np.linalg.norm(a[0]-b[0])
            if d<bd: bd=d;best=i
        if best is not None: used.add(best);pairs.append((a,prev[best]))
    return pairs
def body_pt(c,s,j):
    i=int(s[0,j])
    if i<0 or i>=len(c): return None
    x,y=c[i]; return np.array([x,y]) if np.isfinite(x) and np.isfinite(y) else None

def body_joints(p):
    b=p.get('bodies') if p else None
    if b is None: return {}
    c=np.asarray(b['candidate']); s=np.asarray(b['subset'])
    if s.ndim!=2: return {}
    out={}
    for j in range(min(18,s.shape[1])):
        if j not in BODY_ALLOWED: continue
        i=int(s[0,j])
        if 0<=i<len(c):
            x,y=c[i]
            if np.isfinite(x) and np.isfinite(y): out[j]=np.array([x,y])
    return out

def _hands_labels(p):
    if not p or p.get('hands') is None: return [],[]
    raw=np.asarray(p['hands']); ir=p.get('hands_is_right'); ir=np.asarray(ir) if ir is not None else None
    rw=lw=None
    if ir is None:
        rw=body_pt(np.asarray(p['bodies']['candidate']),np.asarray(p['bodies']['subset']),4) if p.get('bodies') is not None else None
        lw=body_pt(np.asarray(p['bodies']['candidate']),np.asarray(p['bodies']['subset']),7) if p.get('bodies') is not None else None
    H=[];L=[]
    for i,x in enumerate(raw):
        h=valid_hand(x)
        if h is None: continue
        if ir is not None and i<len(ir): lab=bool(ir[i])
        elif rw is not None and lw is not None: lab=np.linalg.norm(h[0]-rw)<np.linalg.norm(h[0]-lw)
        else: lab=None
        H.append(h); L.append(lab)
    return H,L

def per_frame_vals(D, kind):
    """returns (vals, worst, presence); worst=body joint idx; presence=hand appear/disappear."""
    n=len(D); vals=[np.nan]*n; worst=[None]*n; presence=[False]*n
    prevH=prevL=prevkey=None; prevB={}
    for k in range(n):
        p=D[k].get('pose') if isinstance(D[k],dict) else None
        if kind=="hand":
            hs,L=_hands_labels(p)
            key = frozenset(L) if (hs and all(x is not None for x in L) and len(set(L))==len(L)) else ("n",len(hs))
            if k>0 and hs and prevH:
                # label-match if possible, else nearest-root
                vv=[]
                if all(x is not None for x in L) and all(x is not None for x in prevL) and len(set(L))==len(L) and len(set(prevL))==len(prevL):
                    pm={l:h for h,l in zip(prevH,prevL)}
                    for h,l in zip(hs,L):
                        if l in pm:
                            b=pm[l]; vv.append(np.mean(np.linalg.norm(h-b,axis=1))/((handsize(h)+handsize(b))/2))
                else:
                    vv=[np.mean(np.linalg.norm(a-b,axis=1))/((handsize(a)+handsize(b))/2) for a,b in match(hs,prevH)]
                if vv: vals[k]=float(np.median(vv))
            if k>0 and prevkey is not None and key!=prevkey: presence[k]=True
            prevH,prevL,prevkey=hs,L,key
        else:  # body: ALL 18 joints, per-frame value = worst (max) joint jump / shoulder width
            cur=body_joints(p)
            sw = np.linalg.norm(cur[RSHO]-cur[LSHO]) if (RSHO in cur and LSHO in cur) else None
            if sw and sw>0 and prevB:
                per={j:np.linalg.norm(cur[j]-prevB[j])/sw for j in cur if j in prevB}
                if per:
                    wj=max(per,key=per.get); vals[k]=float(per[wj]); worst[k]=wj
            prevB=cur
    return vals, worst, presence

def render(args):
    cid, src = args
    jdir, pkldir, kind, p95k, thr = CFG[src]
    d=pickle.load(open(f"{pkldir}/{cid}_dwpose.pkl",'rb'))
    vals, worst, presence = per_frame_vals(d, kind)
    outdir=f"{BASE}/vis/{src}"; os.makedirs(outdir,exist_ok=True)
    cap=cv2.VideoCapture(f"{VID}/{cid}.mp4"); w=int(cap.get(3)) or 1920; h=int(cap.get(4)) or 1080
    ff=subprocess.Popen(["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pix_fmt","bgr24","-s",f"{w}x{h}",
        "-r","30","-i","pipe:0","-c:v","libx264","-pix_fmt","yuv420p","-crf","22","-movflags","+faststart",
        f"{outdir}/{cid}_mid.mp4"],stdin=subprocess.PIPE)
    idx=0; F=cv2.FONT_HERSHEY_SIMPLEX
    while True:
        ok,img=cap.read()
        if not ok: break
        if (img.shape[1],img.shape[0])!=(w,h): img=cv2.resize(img,(w,h))
        p=d[idx].get('pose') if idx<len(d) else None
        v=vals[idx] if idx<len(vals) else np.nan
        pres=presence[idx] if idx<len(presence) else False
        bad=(np.isfinite(v) and v>thr) or pres
        if p is not None:
            if kind=="hand" and p.get('hands') is not None:
                for hnd in np.asarray(p['hands']):
                    H=valid_hand(hnd)
                    if H is None: continue
                    col=(0,0,255) if bad else (0,255,255)
                    for (x,y) in H: cv2.circle(img,(int(x*w),int(y*h)),3,col,-1)
            elif kind=="body" and p.get('bodies') is not None:
                c=np.asarray(p['bodies']['candidate']); s=np.asarray(p['bodies']['subset'])
                # full skeleton (green), all valid joints as dots; worst joint red on bad frame
                for a,b in BODY_EDGES:
                    pa,pb=body_pt(c,s,a),body_pt(c,s,b)
                    if pa is not None and pb is not None:
                        cv2.line(img,(int(pa[0]*w),int(pa[1]*h)),(int(pb[0]*w),int(pb[1]*h)),(0,200,0),2)
                for j in range(18):
                    if j not in BODY_ALLOWED: continue
                    pj=body_pt(c,s,j)
                    if pj is None: continue
                    isworst = bad and worst[idx]==j
                    cv2.circle(img,(int(pj[0]*w),int(pj[1]*h)),6 if isworst else 3,
                               (0,0,255) if isworst else (0,255,255),-1)
        cv2.rectangle(img,(0,0),(w-1,50),(0,0,0),-1)
        txt=f"f{idx} {idx/FPS:4.1f}s  {src} jump={v:.2f}" if np.isfinite(v) else f"f{idx} {src} jump=-"
        if pres: txt+="  PRESENCE"
        cv2.putText(img,txt+("   BAD" if bad else ""),(12,34),F,0.9,(0,0,255) if bad else (0,255,0),2)
        ff.stdin.write(np.ascontiguousarray(img).tobytes()); idx+=1
    cap.release(); ff.stdin.close(); ff.wait(); return cid

def _readone(args):
    path,p95k=args
    try: r=json.load(open(path))
    except: return None
    if r.get('segments'): return (r['clip'], r.get(p95k) or 0)
    return None
def worst_clips(src, topn):
    jdir, pkldir, kind, p95k, thr = CFG[src]
    D=f"{BASE}/{jdir}"
    files=[(os.path.join(D,f),p95k) for f in os.listdir(D) if f.endswith('.json')]
    with Pool(16) as pool:
        rows=[r for r in pool.map(_readone, files, chunksize=256) if r]
    rows.sort(key=lambda x:-x[1])
    return [c for c,_ in rows[:topn]]

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument('--src',required=True); ap.add_argument('--topn',type=int,default=10)
    a=ap.parse_args()
    cids=worst_clips(a.src, a.topn)
    print(f"{a.src}: rendering {len(cids)} worst -> {BASE}/vis/{a.src}",flush=True)
    with Pool(6) as pool:
        for c in pool.imap_unordered(render,[(c,a.src) for c in cids]): print("  viz",c,flush=True)
    print(f"{a.src}_VIZ_DONE",flush=True)
