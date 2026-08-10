"""Per-clip counts of the found+0-hands categories; write top-10 clips per category."""
import os, pickle, numpy as np
from multiprocessing import Pool
OUT="/home/ubuntu/us-west-3-fs/sahithi/multi_correction/full_run/pkl"
ORIG="/home/ubuntu/us-west-3-fs/sahithi/Wilor/video_cut_wilor_rem"
ARMS={'R':(2,3,4),'L':(5,6,7)}
CATS=['genuine_nohands','far_ok','noarm_nohandnear','suspect_arm','noarm_handnear']

def head(cand,row,W,H):
    for k in (0,1):
        if row[k]>=0: return cand[int(row[k])]*[W,H]
    return None
def arm_eps(cand,row,W,H):
    eps=[]
    for s in ('R','L'):
        sh,el,wr=ARMS[s]
        def pt(i): return cand[int(row[i])]*[W,H] if row[i]>=0 else None
        w,e,spt=pt(wr),pt(el),pt(sh)
        if w is not None: eps.append(w)
        elif e is not None and spt is not None: eps.append(e+(e-spt))
        elif e is not None: eps.append(e)
    return eps

def scan(fn):
    stem=fn[:-len('_dealer.pkl')]
    op=os.path.join(ORIG,f"{stem}_dwpose.pkl")
    if not os.path.exists(op): return None
    try:
        do=pickle.load(open(os.path.join(OUT,fn),'rb')); dO=pickle.load(open(op,'rb'))
    except Exception: return None
    c={k:0 for k in CATS}; ex={k:None for k in CATS}
    for fi,(fo,fO) in enumerate(zip(do,dO)):
        b=fo['pose']['bodies']
        if b is None or np.asarray(fo['pose']['hands']).shape[0]!=0: continue
        bO=fO['pose']['bodies']
        if bO is None: continue
        H,W=fo['frame_dimensions']; diag=np.hypot(W,H)
        hO=np.asarray(fO['pose']['hands']) if fO['pose'].get('hands') is not None else None
        if hO is None or hO.shape[0]==0:
            c['genuine_nohands']+=1
            if ex['genuine_nohands'] is None: ex['genuine_nohands']=fi
            continue
        co=np.asarray(b['candidate']); so=np.asarray(b['subset'])[0]; hsel=head(co,so,W,H)
        if hsel is None: continue
        cO=np.asarray(bO['candidate']); sO=np.asarray(bO['subset'])
        n=min(range(sO.shape[0]), key=lambda mm:(np.hypot(*(head(cO,sO[mm],W,H)-hsel)) if head(cO,sO[mm],W,H) is not None else 1e9))
        eps=arm_eps(cO,sO[n],W,H); wr=hO[:,0,:]*[W,H]
        if eps:
            dmin=min(np.hypot(*(wr[j]-e)) for j in range(len(wr)) for e in eps)
            cat='suspect_arm' if dmin<0.15*diag else 'far_ok'
        else:
            neck=head(cO,sO[n],W,H)
            near=[j for j in range(len(wr)) if np.hypot(*(wr[j]-neck))<0.25*diag and wr[j][1]>neck[1]]
            cat='noarm_handnear' if near else 'noarm_nohandnear'
        c[cat]+=1
        if ex[cat] is None: ex[cat]=fi
    return (stem,c,ex)

if __name__=='__main__':
    files=[f for f in os.listdir(OUT) if f.endswith('.pkl')]
    with Pool(24) as p:
        res=[r for r in p.map(scan,files,chunksize=200) if r]
    sp="/tmp/claude-1000/-home-ubuntu/88cc26b9-b104-4dd1-b365-b88dd225bc5c/scratchpad"
    for cat in CATS:
        top=[t for t in sorted(res,key=lambda t:-t[1][cat]) if t[1][cat]>0][:10]
        with open(f"{sp}/nh_{cat}.txt","w") as fo:
            for stem,cnt,ex in top: fo.write(f"{stem}\t{ex[cat]}\n")
        print(f"{cat}: top10 counts = {[c[cat] for _,c,_ in top]}")
