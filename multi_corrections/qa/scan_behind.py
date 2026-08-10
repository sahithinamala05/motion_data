"""Flag clips where the SELECTED dealer is likely a BEHIND person: in a found frame, the
selected body is much smaller than another body present (a larger/closer person we skipped)."""
import os, pickle, numpy as np
from multiprocessing import Pool
OUT="/home/ubuntu/us-west-3-fs/sahithi/multi_correction/full_run/pkl"
ORIG="/home/ubuntu/us-west-3-fs/sahithi/Wilor/video_cut_wilor_rem"

def head(cand, row, W, H):
    for j in (0,1):
        if row[j] >= 0: return cand[int(row[j])]*[W,H]
    return None
def torso(cand, row, W, H):
    def pt(i): return cand[int(row[i])]*[W,H] if row[i] >= 0 else None
    neck=pt(1)
    if neck is None: neck=pt(0)
    if neck is None: return 1.0
    hips=[p[1] for p in (pt(8),pt(11)) if p is not None]
    if hips: return max(min(hips)-neck[1],1.0)
    rs,ls=pt(2),pt(5)
    return abs(rs[0]-ls[0]) if (rs is not None and ls is not None) else 1.0

def scan(fn):
    stem=fn[:-len('_dealer.pkl')]
    op=os.path.join(ORIG,f"{stem}_dwpose.pkl")
    if not os.path.exists(op): return None
    try:
        do=pickle.load(open(os.path.join(OUT,fn),'rb')); dO=pickle.load(open(op,'rb'))
    except Exception: return None
    found=0; behind=0
    for fo,fO in zip(do,dO):
        b=fo['pose']['bodies']
        if b is None: continue
        bO=fO['pose']['bodies']
        if bO is None: continue
        H,W=fo['frame_dimensions']
        candO=np.asarray(bO['candidate']); subO=np.asarray(bO['subset'])
        if subO.shape[0]<2: continue                      # only one person -> can't be "behind"
        # selected head (from compacted output body)
        co=np.asarray(b['candidate']); so=np.asarray(b['subset'])[0]
        hsel=head(co,so,W,H)
        if hsel is None: continue
        # match to an original body + sizes
        sizes=[]; sel_i=None; bd=1e18
        for n in range(subO.shape[0]):
            hn=head(candO,subO[n],W,H)
            sz=torso(candO,subO[n],W,H); sizes.append(sz)
            if hn is not None:
                dd=np.hypot(*(hn-hsel))
                if dd<bd: bd,sel_i=dd,n
        if sel_i is None: continue
        found+=1
        if sizes[sel_i] < 0.8*max(sizes):                 # a notably larger body exists, not selected
            behind+=1
    if found<40: return None
    return (stem, behind/found, found)

if __name__=='__main__':
    files=[f for f in os.listdir(OUT) if f.endswith('.pkl')]
    with Pool(20) as p:
        res=[r for r in p.map(scan, files, chunksize=200) if r]
    res.sort(key=lambda t:-t[1])
    strong=[r for r in res if r[1]>=0.5]
    with open("/tmp/claude-1000/-home-ubuntu/88cc26b9-b104-4dd1-b365-b88dd225bc5c/scratchpad/behind.txt","w") as fo:
        for stem,r,n in res[:100]: fo.write(stem+"\n")
    print(f"scanned {len(files)}, eligible(>1 body somewhere) {len(res)}")
    print(f"clips with behind-rate >=50%: {len(strong)};  >=30%: {sum(1 for r in res if r[1]>=0.3)}")
    for stem,r,n in res[:8]: print(f"  {stem}  behind={r*100:.0f}%  found={n}")
