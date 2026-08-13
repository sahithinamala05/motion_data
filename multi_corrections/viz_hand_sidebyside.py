"""Side-by-side hand temporal viz: LEFT = WiLoR hands, RIGHT = DWpose hands, same clip/frame.
Each panel draws that tracker's hand keypoints + per-frame jump value + PRESENCE flag, red on BAD
(jump>1.0 OR presence change). Saves <cid>_mid.mp4 to vis/wilor_dwpose_vis/.
"""
import os, sys, pickle, subprocess
import numpy as np, cv2; cv2.setNumThreads(1)
sys.path.insert(0,"/home/ubuntu/us-west-3-fs/sahithi/_entire_code/multi_corrections")
import importlib.util
_s=importlib.util.spec_from_file_location("vt","/home/ubuntu/us-west-3-fs/sahithi/_entire_code/multi_corrections/viz_temporal.py")
vt=importlib.util.module_from_spec(_s); _s.loader.exec_module(vt)

VC="/home/ubuntu/us-west-3-fs/sahithi/Wilor/video_cut_wilor"
DW="/home/ubuntu/us-west-3-fs/live_dealer_blackjack/dwpose/batch_01/dwpose"
VID="/home/ubuntu/us-west-3-fs/live_dealer_blackjack/video_cut/batch_01"
OUT="/home/ubuntu/us-west-3-fs/sahithi/hand_consistency/vis/wilor_dwpose_vis"
FPS=30.0; THR=1.0
F=cv2.FONT_HERSHEY_SIMPLEX

def draw_panel(img, pose, val, pres, tag):
    bad=(np.isfinite(val) and val>THR) or pres
    if pose is not None and pose.get('hands') is not None:
        for hnd in np.asarray(pose['hands']):
            H=vt.valid_hand(hnd)
            if H is None: continue
            col=(0,0,255) if bad else (0,255,255)
            for (x,y) in H: cv2.circle(img,(int(x*img.shape[1]),int(y*img.shape[0])),3,col,-1)
    cv2.rectangle(img,(0,0),(img.shape[1]-1,50),(0,0,0),-1)
    t=f"{tag}  jump={val:.2f}" if np.isfinite(val) else f"{tag}  jump=-"
    if pres: t+="  PRESENCE"
    if bad: t+="  BAD"
    cv2.putText(img,t,(12,34),F,0.8,(0,0,255) if bad else (0,255,0),2)
    return img

def render(cid):
    W=pickle.load(open(f"{VC}/{cid}_dwpose.pkl",'rb'))
    D=pickle.load(open(f"{DW}/{cid}_dwpose.pkl",'rb'))
    wv,_,wp=vt.per_frame_vals(W,"hand")
    dv,_,dp=vt.per_frame_vals(D,"hand")
    cap=cv2.VideoCapture(f"{VID}/{cid}.mp4"); w=int(cap.get(3)) or 1920; h=int(cap.get(4)) or 1080
    ff=subprocess.Popen(["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pix_fmt","bgr24","-s",f"{2*w}x{h}",
        "-r","30","-i","pipe:0","-c:v","libx264","-pix_fmt","yuv420p","-crf","22","-movflags","+faststart",
        f"{OUT}/{cid}_mid.mp4"],stdin=subprocess.PIPE)
    idx=0
    while True:
        ok,img=cap.read()
        if not ok: break
        if (img.shape[1],img.shape[0])!=(w,h): img=cv2.resize(img,(w,h))
        left=img.copy(); right=img.copy()
        draw_panel(left,  W[idx].get('pose') if idx<len(W) else None, wv[idx] if idx<len(wv) else np.nan, wp[idx] if idx<len(wp) else False, "WiLoR")
        draw_panel(right, D[idx].get('pose') if idx<len(D) else None, dv[idx] if idx<len(dv) else np.nan, dp[idx] if idx<len(dp) else False, "DWpose")
        cv2.putText(left, f"f{idx} {idx/FPS:4.1f}s",(w-260,34),F,0.8,(255,255,255),2)
        ff.stdin.write(np.ascontiguousarray(np.hstack([left,right])).tobytes()); idx+=1
    cap.release(); ff.stdin.close(); ff.wait(); return cid

if __name__=="__main__":
    os.makedirs(OUT,exist_ok=True)
    src="/home/ubuntu/us-west-3-fs/sahithi/hand_consistency/vis/wilor"
    cids=sorted(f[:-len('_mid.mp4')] for f in os.listdir(src) if f.endswith('_mid.mp4'))
    print(f"side-by-side rendering {len(cids)} clips -> {OUT}",flush=True)
    for c in cids:
        try: render(c); print("  viz",c,flush=True)
        except Exception as e: print("  ERR",c,e,flush=True)
    print("DONE",flush=True)
