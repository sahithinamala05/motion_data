"""Visualize FACT ckpt diff: old (iter-40000) vs latest segmentation over the video.
Two timeline strips (top=OLD, bottom=NEW) with a moving cursor; per-frame active label
for each; text turns red where OLD != NEW. Reads clip\tvidpath from diff_viz.txt.
"""
import os, sys, json, subprocess
import numpy as np, cv2; cv2.setNumThreads(1)
OLD="/home/ubuntu/us-west-3-fs/sahithi/initial_hand_ckpt_diff/pred_40000/predictions_json"
# NEW = autolabeling union (part_1 + remaining), overridable via $NEW_DIRS (colon-separated)
NEW_DIRS=os.environ.get("NEW_DIRS",
    "/home/ubuntu/us-west-3-fs/live_dealer_blackjack/action_annotation/mixed_mini_batch/annotation/AutoLabeling_batch_01_part_1:"
    "/home/ubuntu/us-west-3-fs/live_dealer_blackjack/action_annotation/mixed_mini_batch/annotation/Autolabeling_remaining").split(":")
def new_path(cid):
    for d in NEW_DIRS:
        p=f"{d}/{cid}_predictions.json"
        if os.path.exists(p): return p
    return None
OUT=os.environ.get("VIZ_OUT","/home/ubuntu/us-west-3-fs/sahithi/initial_hand_ckpt_diff/vis_diff")
FPS=30.0; F=cv2.FONT_HERSHEY_SIMPLEX
COL={"background":(90,90,90),"clean hand":(120,255,120),"close bets":(200,120,255),
     "discard":(180,180,0),"initial hands 1st":(255,150,0),"initial hands 2nd":(255,90,0),
     "reveal hole card":(255,0,255)}

def frames_arr(path):
    segs=json.load(open(path))["timeline_segments"]
    if not segs: return np.array([],dtype=object)
    N=max(s["end_frame"] for s in segs)+1
    a=np.array(["background"]*N,dtype=object)
    for s in segs:
        a[s["start_frame"]:s["end_frame"]+1]= s["labels"][0] if s["labels"] else "background"
    return a

def strip(img, arr, y0, y1, N):
    w=img.shape[1]
    for i in range(N):
        c=COL.get(arr[i] if i<len(arr) else "background",(90,90,90))
        x=int(w*i/N); cv2.line(img,(x,y0),(x,y1),c,1)

def render(cid, vp):
    A=frames_arr(f"{OLD}/{cid}.json"); B=frames_arr(new_path(cid))
    cap=cv2.VideoCapture(vp); w=int(cap.get(3)) or 1920; h=int(cap.get(4)) or 1080
    N=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or max(len(A),len(B))
    ff=subprocess.Popen(["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pix_fmt","bgr24","-s",f"{w}x{h}",
        "-r","30","-i","pipe:0","-c:v","libx264","-pix_fmt","yuv420p","-crf","22","-movflags","+faststart",
        f"{OUT}/{cid}_mid.mp4"],stdin=subprocess.PIPE)
    idx=0
    while True:
        ok,img=cap.read()
        if not ok: break
        if (img.shape[1],img.shape[0])!=(w,h): img=cv2.resize(img,(w,h))
        la=A[idx] if idx<len(A) else "background"; lb=B[idx] if idx<len(B) else "background"
        diff=la!=lb
        cv2.rectangle(img,(0,0),(w-1,150),(0,0,0),-1)
        cv2.putText(img,f"f{idx} {idx/FPS:4.1f}s",(12,28),F,0.7,(255,255,255),2)
        cv2.putText(img,f"OLD (iter40000): {la}",(12,58),F,0.8,(0,0,255) if diff else (0,255,0),2)
        cv2.putText(img,f"NEW (latest)   : {lb}",(12,88),F,0.8,(0,0,255) if diff else (0,255,0),2)
        strip(img,A,105,120,N); strip(img,B,128,143,N)
        cx=int(w*idx/max(N,1))
        cv2.line(img,(cx,103),(cx,145),(255,255,255),1)
        cv2.putText(img,"OLD",(w-120,117),F,0.5,(255,255,255),1); cv2.putText(img,"NEW",(w-120,140),F,0.5,(255,255,255),1)
        ff.stdin.write(np.ascontiguousarray(img).tobytes()); idx+=1
    cap.release(); ff.stdin.close(); ff.wait(); return cid

if __name__=="__main__":
    os.makedirs(OUT,exist_ok=True)
    for line in open(os.environ.get("DIFF_LIST","/tmp/claude-1000/-home-ubuntu/6938b38c-0207-4beb-ac65-f66c4fab8367/scratchpad/diff_viz.txt")):
        cid,vp=line.strip().split("\t")
        try: render(cid,vp); print("  viz",cid,flush=True)
        except Exception as e: print("  ERR",cid,e,flush=True)
    print("DONE",flush=True)
