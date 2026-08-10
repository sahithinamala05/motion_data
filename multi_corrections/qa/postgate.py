"""Apply stricter gates to existing dealer pkls (drop-only): a frame's dealer is kept only if
its DWPose face has >=FACE_KPT_MIN valid keypoints AND its body has >=MIN_BODY_KPTS joints.
Equivalent to re-running with the new thresholds, since both gates only drop frames."""
import os, pickle, numpy as np
from multiprocessing import Pool
PKL="/home/ubuntu/us-west-3-fs/sahithi/multi_correction/full_run/pkl"
FACE_KPT_MIN=66; MIN_BODY_KPTS=8
EMPTY_HANDS=np.zeros((0,21,2))

def valid_face(fa):
    if fa is None: return 0
    fa=np.asarray(fa)
    if fa.ndim!=3 or fa.shape[0]==0: return 0
    f=fa[0]; return int(((f>=0)&(f<=1)).all(axis=1).sum())

def process(fn):
    p=os.path.join(PKL,fn)
    try: d=pickle.load(open(p,'rb'))
    except Exception: return (0,0)
    flipped=0; found=0
    for fr in d:
        b=fr['pose']['bodies']
        if b is None: continue
        found+=1
        sub=np.asarray(b['subset'])
        bad = valid_face(fr['pose'].get('faces')) < FACE_KPT_MIN or int((sub[0]>=0).sum()) < MIN_BODY_KPTS
        if bad:
            fr['pose']={'bodies':None,'hands':EMPTY_HANDS,'hands_is_right':[],'faces':None}
            fr['dealer_not_found']=True
            flipped+=1
    if flipped:
        pickle.dump(d,open(p,'wb'))
    return (found,flipped)

if __name__=='__main__':
    files=[f for f in os.listdir(PKL) if f.endswith('.pkl')]
    with Pool(20) as pool:
        res=pool.map(process, files, chunksize=200)
    F=sum(a for a,_ in res); X=sum(b for _,b in res)
    print(f"pkls={len(files)}  dealer-found frames before={F}  flipped->not_found={X}  ({100*X/max(F,1):.2f}%)")
