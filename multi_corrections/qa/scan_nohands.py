"""For every frame where the dealer IS found but 0 hands were kept, decide whether that's
legitimate (no visible dealer hand) or a wrong drop (a WiLoR hand near the dealer's arm exists)."""
import os, pickle, numpy as np
from multiprocessing import Pool
OUT="/home/ubuntu/us-west-3-fs/sahithi/multi_correction/full_run/pkl"
ORIG="/home/ubuntu/us-west-3-fs/sahithi/Wilor/video_cut_wilor_rem"
ARMS={'R':(2,3,4),'L':(5,6,7)}

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
    c=dict(genuine_nohands=0, far_ok=0, suspect_arm=0, noarm_nohandnear=0, noarm_handnear=0)
    for fo,fO in zip(do,dO):
        b=fo['pose']['bodies']
        if b is None: continue                          # dealer not found -> not a 'found+0hands' frame
        if np.asarray(fo['pose']['hands']).shape[0]!=0: continue   # hands were kept -> skip
        bO=fO['pose']['bodies']
        if bO is None: continue
        H,W=fo['frame_dimensions']; diag=np.hypot(W,H)
        hO=np.asarray(fO['pose']['hands']) if fO['pose'].get('hands') is not None else None
        if hO is None or hO.shape[0]==0:
            c['genuine_nohands']+=1; continue           # no WiLoR hands in frame at all
        # recover dealer body in original by head match
        co=np.asarray(b['candidate']); so=np.asarray(b['subset'])[0]; hsel=head(co,so,W,H)
        if hsel is None: continue
        cO=np.asarray(bO['candidate']); sO=np.asarray(bO['subset'])
        n=min(range(sO.shape[0]), key=lambda mm:(np.hypot(*(head(cO,sO[mm],W,H)-hsel)) if head(cO,sO[mm],W,H) is not None else 1e9))
        eps=arm_eps(cO,sO[n],W,H)
        wr=hO[:,0,:]*[W,H]
        if eps:
            dmin=min(np.hypot(*(wr[j]-e)) for j in range(len(wr)) for e in eps)
            if dmin < 0.15*diag: c['suspect_arm']+=1     # a hand sits on the dealer's arm but wasn't kept
            else: c['far_ok']+=1                          # hands exist but far -> other people's
        else:
            # dealer's arm undetected -> can't link. is a hand plausibly hers (near her neck, below it)?
            neck=head(cO,sO[n],W,H)
            near=[j for j in range(len(wr)) if np.hypot(*(wr[j]-neck))<0.25*diag and wr[j][1]>neck[1]]
            if near: c['noarm_handnear']+=1
            else: c['noarm_nohandnear']+=1
    return c

if __name__=='__main__':
    files=[f for f in os.listdir(OUT) if f.endswith('.pkl')]
    with Pool(24) as p:
        res=[r for r in p.map(scan,files,chunksize=200) if r]
    tot={}
    for r in res:
        for k,v in r.items(): tot[k]=tot.get(k,0)+v
    T=sum(tot.values())
    print(f"found+0-hands frames analysed: {T}")
    for k in ('genuine_nohands','far_ok','noarm_nohandnear','suspect_arm','noarm_handnear'):
        print(f"  {k:18s}: {tot.get(k,0):8d}  ({100*tot.get(k,0)/max(T,1):.2f}%)")
    print(f"\nLEGIT (no visible dealer hand) = genuine + far_ok + noarm_nohandnear = {tot['genuine_nohands']+tot['far_ok']+tot['noarm_nohandnear']}")
    print(f"SUSPECT (a hand near the dealer was not kept) = suspect_arm + noarm_handnear = {tot['suspect_arm']+tot['noarm_handnear']}")
