"""Find clips where the 'behind-person' negative-evidence branch in link_hands actually fired
(a per-wrist nearest hand was dropped because it was < OTHER_MARGIN * closer to another person's arm)."""
import os, pickle, numpy as np
from multiprocessing import Pool
OUT="/home/ubuntu/us-west-3-fs/sahithi/multi_correction/full_run/pkl"
ORIG="/home/ubuntu/us-west-3-fs/sahithi/Wilor/video_cut_wilor_rem"
ARMS={'R':(2,3,4),'L':(5,6,7)}; OTHER_MARGIN=0.8

def arm_endpoint(cand,row,side,W,H):
    sh,el,wr=ARMS[side]
    def pt(i): return cand[int(row[i])]*[W,H] if row[i]>=0 else None
    w,e,s=pt(wr),pt(el),pt(sh)
    if w is not None: return w
    if e is not None and s is not None: return e+(e-s)
    if e is not None: return e
    return None

def head(cand,row,W,H):
    for k in (0,1):
        if row[k]>=0: return cand[int(row[k])]*[W,H]
    return None

def fired(cand,sub,hands,n,W,H):
    """True if negative-evidence would DROP a per-wrist-nearest hand for dealer body n."""
    targets={s:arm_endpoint(cand,sub[n],s,W,H) for s in ('R','L')}
    targets={s:ep for s,ep in targets.items() if ep is not None}
    if not targets or hands is None or hands.shape[0]==0: return False
    others=[]
    for m in range(sub.shape[0]):
        if m==n: continue
        for s in ('R','L'):
            ep=arm_endpoint(cand,sub[m],s,W,H)
            if ep is not None: others.append(ep)
    if not others: return False
    wr=hands[:,0,:]*[W,H]
    nearest={s:min(range(len(wr)),key=lambda j:np.hypot(*(wr[j]-ep))) for s,ep in targets.items()}
    if nearest.get('R')==nearest.get('L'):
        s=min(targets,key=lambda s:np.hypot(*(wr[nearest[s]]-targets[s]))); nearest={s:nearest[s]}
    for s,j in nearest.items():
        dd=np.hypot(*(wr[j]-targets[s]))
        if min(np.hypot(*(wr[j]-eo)) for eo in others) < OTHER_MARGIN*dd:
            return True
    return False

def scan(fn):
    stem=fn[:-len('_dealer.pkl')]
    op=os.path.join(ORIG,f"{stem}_dwpose.pkl")
    if not os.path.exists(op): return None
    try:
        do=pickle.load(open(os.path.join(OUT,fn),'rb')); dO=pickle.load(open(op,'rb'))
    except Exception: return None
    cnt=0
    for fo,fO in zip(do,dO):
        b=fo['pose']['bodies']
        if b is None: continue                        # dealer not found this frame
        bO=fO['pose']['bodies']
        if bO is None: continue
        H,W=fo['frame_dimensions']
        co=np.asarray(b['candidate']); so=np.asarray(b['subset'])[0]
        hsel=head(co,so,W,H)
        if hsel is None: continue
        cO=np.asarray(bO['candidate']); sO=np.asarray(bO['subset'])
        if sO.shape[0]<2: continue                    # need a 'behind' person for negatives to exist
        # recover dealer body index in ORIGINAL by head match
        n=min(range(sO.shape[0]), key=lambda m:(np.hypot(*(head(cO,sO[m],W,H)-hsel)) if head(cO,sO[m],W,H) is not None else 1e9))
        hands=np.asarray(fO['pose']['hands']) if fO['pose'].get('hands') is not None else None
        if fired(cO,sO,hands,n,W,H): cnt+=1
    return (stem,cnt) if cnt>0 else None

if __name__=='__main__':
    files=[f for f in os.listdir(OUT) if f.endswith('.pkl')]
    with Pool(24) as p:
        res=[r for r in p.map(scan,files,chunksize=200) if r]
    res.sort(key=lambda t:-t[1])
    tot=sum(c for _,c in res)
    print(f"clips where negative-evidence fired at least once: {len(res)}  (total frames affected: {tot})")
    print("top clips (stem, frames):")
    for s,c in res[:15]: print(f"  {s}  {c}")
    sp="/tmp/claude-1000/-home-ubuntu/88cc26b9-b104-4dd1-b365-b88dd225bc5c/scratchpad"
    open(f"{sp}/negev.txt","w").write("\n".join(s for s,_ in res))
