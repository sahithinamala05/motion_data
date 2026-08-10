"""Rank multi-dealer output pkls by how often the dealer's DWPose WRIST keypoints are missing."""
import os, pickle, numpy as np
from multiprocessing import Pool
PKL="/home/ubuntu/us-west-3-fs/sahithi/multi_correction/full_run/pkl"

def rate(fn):
    try:
        d=pickle.load(open(os.path.join(PKL,fn),'rb'))
    except Exception:
        return None
    found=0; miss=0
    for f in d:
        b=f['pose']['bodies']
        if b is None: continue
        sub=np.asarray(b['subset'])[0]
        found+=1
        if sub[4]<0 or sub[7]<0:          # R-wrist(4) or L-wrist(7) missing
            miss+=1
    if found<50: return None              # need enough dealer frames to be meaningful
    return (fn[:-len('_dealer.pkl')], miss/found, found)

if __name__=='__main__':
    files=[f for f in os.listdir(PKL) if f.endswith('.pkl')]
    with Pool(20) as p:
        res=[r for r in p.map(rate, files, chunksize=200) if r]
    res.sort(key=lambda t:-t[1])
    top=res[:100]
    with open("/tmp/claude-1000/-home-ubuntu/88cc26b9-b104-4dd1-b365-b88dd225bc5c/scratchpad/wrist100.txt","w") as fo:
        for stem,r,n in top: fo.write(stem+"\n")
    print(f"scanned {len(files)} pkls, {len(res)} eligible")
    print(f"top-100 wrist-missing rate: {top[0][1]*100:.0f}% .. {top[-1][1]*100:.0f}%")
    print("examples:")
    for stem,r,n in top[:5]: print(f"  {stem}  miss={r*100:.0f}%  found={n}")
    # distribution
    rr=np.array([r for _,r,_ in res])
    for lo,hi in [(0,.1),(.1,.3),(.3,.6),(.6,1.01)]:
        print(f"  clips wrist-miss {int(lo*100)}-{int(hi*100)}%: {((rr>=lo)&(rr<hi)).sum()}")
