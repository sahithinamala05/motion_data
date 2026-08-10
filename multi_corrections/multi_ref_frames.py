"""One frontal, center-dealer-cropped frame per local dealer id, sourced from that
dealer's MULTI-person clips. Output: 70k_multi/<dealer_id>.png (resumable).
Usage: python multi_ref_frames.py [--limit N] [--ids d1,d2,...]
"""
import os, argparse
import numpy as np, cv2, pandas as pd
from insightface.app import FaceAnalysis

VC="/home/ubuntu/us-west-3-fs/live_dealer_blackjack/video_cut/batch_01"
PARQUET="/home/ubuntu/us-west-3-fs/sahithi/db/db_batch_01.parquet"
OUT="/home/ubuntu/us-west-3-fs/sahithi/multi_correction/70k_multi"
N_SAMPLE=25            # frames sampled per clip to hunt for a frontal one
FACE_WIN=3.0           # half-width of crop = FACE_WIN * face_width  (total ~6x)
SHARP_MIN=60.0         # min variance-of-Laplacian on the face crop (rejects blurred shuffle/transition frames)
GOOD_CAM=-12.0         # cam_facing good enough to stop searching (|pitch|+|yaw| <= 12 deg -> looking at lens)
MAX_CLIPS=3            # clips to search per dealer for a good look-up frame

def cam_facing(f):
    """higher = face pointed more directly at the camera (looking up/at lens, not down).
    pose=[pitch,yaw,roll]; down-turned frames have large-negative pitch -> heavily penalized."""
    p=getattr(f,'pose',None)
    if p is None: return -1e9
    pitch,yaw=float(p[0]),float(p[1])
    return -(abs(yaw)+abs(pitch))

def center_face(faces, W):
    """The center dealer's face = large AND central (area * centrality)."""
    best,bs=None,-1
    for f in faces:
        a=(f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])
        cx=(f.bbox[0]+f.bbox[2])/2
        centr=1.0-abs(cx/W-0.5)/0.5
        s=a*max(centr,0.05)
        if s>bs: bs,best=s,f
    return best


def sharpness(img, bbox):
    """variance of Laplacian on the face crop; low = blurred (shuffle/transition/motion)."""
    x0,y0,x1,y1=[int(v) for v in bbox]
    x0,y0=max(0,x0),max(0,y0)
    fc=img[y0:y1, x0:x1]
    if fc.size==0: return 0.0
    return float(cv2.Laplacian(cv2.cvtColor(fc,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var())

def crop_center(img, f):
    H,W=img.shape[:2]
    cx=(f.bbox[0]+f.bbox[2])/2; fw=f.bbox[2]-f.bbox[0]
    half=FACE_WIN*fw
    x0=int(max(0,cx-half)); x1=int(min(W,cx+half))
    return img[:, x0:x1]                       # horizontal window, full height

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--ids',default='')
    ap.add_argument('--shard',type=int,default=0)     # this worker's index
    ap.add_argument('--nshards',type=int,default=1)   # total workers (ids split round-robin)
    a=ap.parse_args()
    os.makedirs(OUT,exist_ok=True)

    df=pd.read_parquet(PARQUET,columns=['base_name','single_person','local_dealer_id','face_passed'])
    multi=df[df['single_person']==False].dropna(subset=['local_dealer_id'])
    ids=sorted(multi['local_dealer_id'].unique())
    if a.ids: ids=[i for i in a.ids.split(',')]
    if a.limit: ids=ids[:a.limit]
    if a.nshards>1: ids=[d for i,d in enumerate(ids) if i%a.nshards==a.shard]   # round-robin shard

    app=FaceAnalysis(name='buffalo_l',providers=['CUDAExecutionProvider','CPUExecutionProvider'])
    app.prepare(ctx_id=0,det_size=(640,640))

    ok=miss=skip=0
    for k,did in enumerate(ids):
        outp=os.path.join(OUT,f"{did}.png")
        if os.path.exists(outp): skip+=1; continue
        clips=multi[multi['local_dealer_id']==did].sort_values('face_passed',ascending=False)['base_name'].tolist()
        best_pick=None; best_cam=-1e9; prov=None   # best sharp+facing crop; prov=least-blurry fallback
        for bn in clips[:MAX_CLIPS]:
            mp=os.path.join(VC,bn)
            if not os.path.exists(mp): continue
            cap=cv2.VideoCapture(mp); N=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            cands=[]                           # (cam_facing, sharpness, crop)
            for fi in np.linspace(0,max(N-1,0),N_SAMPLE,dtype=int):
                cap.set(cv2.CAP_PROP_POS_FRAMES,int(fi)); r,img=cap.read()
                if not r: continue
                faces=app.get(img)
                if not faces: continue
                cf=center_face(faces,img.shape[1])
                cands.append((cam_facing(cf), sharpness(img,cf.bbox), crop_center(img,cf)))
            cap.release()
            for c in cands:                    # accumulate best camera-facing among SHARP frames across clips
                if c[1]>=SHARP_MIN and c[0]>best_cam:
                    best_cam,best_pick=c[0],c[2]
            if prov is None and cands:
                prov=max(cands,key=lambda c:c[1])[2]
            if best_pick is not None and best_cam>=GOOD_CAM:
                break                          # found a genuine look-up frame -> done
        pick=best_pick if best_pick is not None else prov
        if pick is not None:
            cv2.imwrite(outp,pick); ok+=1
        else:
            miss+=1; print(f"  MISS {did}",flush=True)
        if (k+1)%50==0:
            print(f"{k+1}/{len(ids)}  ok={ok} miss={miss} skip={skip}",flush=True)
    print(f"DONE: ok={ok} miss={miss} skip={skip} -> {OUT}",flush=True)

if __name__=='__main__':
    main()
