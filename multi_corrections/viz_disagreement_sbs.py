"""Side-by-side viz of worst DWpose-vs-WiLoR disagreement clips.
LEFT = WiLoR hands, RIGHT = DWpose hands (same frame). Each panel also draws the OTHER
model's hand faintly so you can SEE the offset. Per-frame disagreement value burned in;
frames with disagreement > 0.5 hand-width flagged red BAD. Saves to vis/model_dis/.
"""
import os, sys, json, pickle, subprocess
import numpy as np, cv2; cv2.setNumThreads(1)
sys.path.insert(0,"/home/ubuntu/us-west-3-fs/sahithi/_entire_code/multi_corrections")
import importlib.util
_s=importlib.util.spec_from_file_location("md","/home/ubuntu/us-west-3-fs/sahithi/_entire_code/multi_corrections/model_disagreement.py")
md=importlib.util.module_from_spec(_s); _s.loader.exec_module(md)

VC="/home/ubuntu/us-west-3-fs/sahithi/Wilor/video_cut_wilor"
DW="/home/ubuntu/us-west-3-fs/live_dealer_blackjack/dwpose/batch_01/dwpose"
VID="/home/ubuntu/us-west-3-fs/live_dealer_blackjack/video_cut/batch_01"
JD="/home/ubuntu/us-west-3-fs/sahithi/hand_consistency/model_disagreement"
OUT="/home/ubuntu/us-west-3-fs/sahithi/hand_consistency/vis/model_dis"
FPS=30.0; THR=0.5; F=cv2.FONT_HERSHEY_SIMPLEX
WCOL=(0,255,255); DCOL=(255,255,0)   # WiLoR yellow, DWpose cyan

def frame_disagree(wp,dp):
    wh=md.sided(wp,derive=False); dh=md.sided(dp,derive=True)
    per=[]
    for side in (True,False):
        if side in wh and side in dh:
            a,b=wh[side],dh[side]; sc=(md.hsz(a)+md.hsz(b))/2
            per.append(float((np.linalg.norm(a-b,axis=1)/sc).mean()))
    return float(np.median(per)) if per else np.nan

def draw_hands(img, pose, col, derive, thick=-1, rad=3):
    hs=md.sided(pose, derive=derive)
    for h in hs.values():
        for (x,y) in h: cv2.circle(img,(int(x*img.shape[1]),int(y*img.shape[0])),rad,col,thick)

def render(cid):
    W=pickle.load(open(f"{VC}/{cid}_dwpose.pkl",'rb')); D=pickle.load(open(f"{DW}/{cid}_dwpose.pkl",'rb'))
    cap=cv2.VideoCapture(f"{VID}/{cid}.mp4"); w=int(cap.get(3)) or 1920; h=int(cap.get(4)) or 1080
    ff=subprocess.Popen(["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pix_fmt","bgr24","-s",f"{2*w}x{h}",
        "-r","30","-i","pipe:0","-c:v","libx264","-pix_fmt","yuv420p","-crf","22","-movflags","+faststart",
        f"{OUT}/{cid}_mid.mp4"],stdin=subprocess.PIPE)
    idx=0
    while True:
        ok,img=cap.read()
        if not ok: break
        if (img.shape[1],img.shape[0])!=(w,h): img=cv2.resize(img,(w,h))
        wp=W[idx].get('pose') if idx<len(W) else None; dp=D[idx].get('pose') if idx<len(D) else None
        dv=frame_disagree(wp,dp); bad=np.isfinite(dv) and dv>THR
        left=img.copy(); right=img.copy()
        # left: WiLoR solid + DWpose faint ; right: DWpose solid + WiLoR faint
        draw_hands(left, dp, (120,120,120), True, thick=-1, rad=2); draw_hands(left, wp, WCOL, False, rad=3)
        draw_hands(right, wp, (120,120,120), False, thick=-1, rad=2); draw_hands(right, dp, DCOL, True, rad=3)
        for panel,tag,col in ((left,"WiLoR",WCOL),(right,"DWpose",DCOL)):
            cv2.rectangle(panel,(0,0),(w-1,50),(0,0,0),-1)
            t=f"{tag}  disagree={dv:.2f}" if np.isfinite(dv) else f"{tag}  disagree=-"
            if bad: t+="  BAD"
            cv2.putText(panel,t,(12,34),F,0.8,(0,0,255) if bad else (0,255,0),2)
        cv2.putText(left,f"f{idx} {idx/FPS:4.1f}s",(w-250,34),F,0.7,(255,255,255),2)
        ff.stdin.write(np.ascontiguousarray(np.hstack([left,right])).tobytes()); idx+=1
    cap.release(); ff.stdin.close(); ff.wait(); return cid

def _sel(p):
    try: r=json.load(open(p))
    except: return None
    return (r['clip'], r.get('disagree_p95') or 0) if r.get('segments') else None

if __name__=="__main__":
    from multiprocessing import Pool
    os.makedirs(OUT,exist_ok=True)
    files=[os.path.join(JD,f) for f in os.listdir(JD) if f.endswith('.json')]
    with Pool(16) as pool:
        rows=[r for r in pool.map(_sel, files, chunksize=256) if r]
    rows.sort(key=lambda x:-x[1]); cids=[c for c,_ in rows[:10]]
    print("worst-disagreement clips:", cids, flush=True)
    for c in cids:
        try: render(c); print("  viz",c,flush=True)
        except Exception as e: print("  ERR",c,e,flush=True)
    print("DONE",flush=True)
