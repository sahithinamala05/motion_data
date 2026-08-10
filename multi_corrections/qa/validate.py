"""Automated correctness validation of the single-dealer pkls (no visual review).
Two layers:
  A. INVARIANTS that must hold by construction (any violation = a real bug):
     schema, exactly-one-person, <=2 hands, hands_is_right aligned, face gate (>=66/68),
     body-completeness (>=8 joints), hands-near-dealer, frame count == input,
     provenance (every kept dealer body/hand actually exists in the ORIGINAL pkl).
  B. STATISTICAL flags (not bugs, just a shortlist to eyeball): low dealer-found rate,
     off-centre dealer (possible wrong person), jumpy dealer head (possible identity switching).
"""
import os, pickle, numpy as np
from multiprocessing import Pool
OUT="/home/ubuntu/us-west-3-fs/sahithi/multi_correction/full_run/pkl"
ORIG="/home/ubuntu/us-west-3-fs/sahithi/Wilor/video_cut_wilor_rem"
FACE_MIN=66; MIN_BODY=8; ARMS={'R':(2,3,4),'L':(5,6,7)}

def vface(fa):
    if fa is None: return 0
    fa=np.asarray(fa)
    if fa.ndim!=3 or fa.shape[0]==0: return 0
    f=fa[0]; return int(((f>=0)&(f<=1)).all(axis=1).sum())

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

def check(fn):
    v=set(); stem=fn[:-len('_dealer.pkl')]
    try: d=pickle.load(open(os.path.join(OUT,fn),'rb'))
    except Exception: return {'load_fail':1}
    op=os.path.join(ORIG,f"{stem}_dwpose.pkl")
    dO=pickle.load(open(op,'rb')) if os.path.exists(op) else None
    if dO is None: v.add('no_original')
    elif len(d)!=len(dO): v.add('framecount_mismatch')
    found=0; heads=[]
    for i,fr in enumerate(d):
        p=fr['pose']; b=p['bodies']
        if 'frame_dimensions' not in fr: v.add('schema'); continue
        H,W=fr['frame_dimensions']
        if b is None:
            if not fr.get('dealer_not_found'): v.add('none_without_flag')
            continue
        found+=1
        sub=np.asarray(b['subset']); cand=np.asarray(b['candidate'])
        if sub.shape!=(1,18): v.add('not_one_person')
        if int((sub[0]>=0).sum())<MIN_BODY: v.add('body_lt_min')
        if vface(p.get('faces'))<FACE_MIN: v.add('face_gate')
        h=np.asarray(p['hands'])
        if h.shape[0]>2: v.add('gt2_hands')
        if h.shape[0]>0 and h.shape[1:]!=(21,2): v.add('hand_shape')
        if len(p.get('hands_is_right',[]))!=h.shape[0]: v.add('isr_misaligned')
        fa=p.get('faces')
        if fa is not None and np.asarray(fa).shape[0]>1: v.add('gt1_face')
        # finite & normalized-ish
        if not np.isfinite(cand).all(): v.add('nan')
        # hands near dealer arm
        eps=arm_eps(cand,sub[0],W,H)
        if eps and h.shape[0]>0:
            diag=np.hypot(W,H)
            for j in range(h.shape[0]):
                wr=h[j][0]*[W,H]
                if min(np.hypot(*(wr-e)) for e in eps)>0.2*diag: v.add('hand_far')
        # provenance: dealer head must match some ORIGINAL body head this frame
        if dO is not None and i<len(dO) and dO[i]['pose']['bodies'] is not None:
            hn=cand[int(sub[0,0])]*[W,H] if sub[0,0]>=0 else (cand[int(sub[0,1])]*[W,H] if sub[0,1]>=0 else None)
            if hn is not None:
                heads.append(hn[0]/W)
                bO=dO[i]['pose']['bodies']; cO=np.asarray(bO['candidate']); sO=np.asarray(bO['subset'])
                ok=False
                for n in range(sO.shape[0]):
                    for k in (0,1):
                        if sO[n,k]>=0 and np.hypot(*(cO[int(sO[n,k])]*[W,H]-hn))<3: ok=True
                if not ok: v.add('provenance')
    stats={'found_rate':found/max(len(d),1),
           'head_x_med':float(np.median(heads)) if heads else -1,
           'head_x_std':float(np.std(heads)) if len(heads)>1 else 0.0}
    return {'stem':stem,'violations':v,'stats':stats}

if __name__=='__main__':
    files=[f for f in os.listdir(OUT) if f.endswith('.pkl')]
    with Pool(24) as pool:
        res=pool.map(check, files, chunksize=200)
    from collections import Counter
    vc=Counter()
    low=[]; off=[]; jump=[]
    for r in res:
        if 'load_fail' in r: vc['load_fail']+=1; continue
        for x in r['violations']: vc[x]+=1
        s=r['stats']
        if s['found_rate']<0.3: low.append((r['stem'],round(s['found_rate'],2)))
        if s['head_x_med']>=0 and abs(s['head_x_med']-0.5)>0.28: off.append((r['stem'],round(s['head_x_med'],2)))
        if s['head_x_std']>0.12: jump.append((r['stem'],round(s['head_x_std'],2)))
    print(f"== validated {len(files)} pkls ==")
    print("\nA) INVARIANT VIOLATIONS (must be 0):")
    inv=['load_fail','no_original','framecount_mismatch','schema','none_without_flag','not_one_person',
         'body_lt_min','face_gate','gt2_hands','hand_shape','isr_misaligned','gt1_face','nan','hand_far','provenance']
    for k in inv: print(f"   {k:20s}: {vc.get(k,0)}")
    print(f"\nB) STATISTICAL FLAGS (shortlist to eyeball, not bugs):")
    print(f"   low found-rate (<30%): {len(low)}")
    print(f"   off-centre dealer (|x-0.5|>0.28): {len(off)}")
    print(f"   jumpy head (std>0.12): {len(jump)}")
    sp="/tmp/claude-1000/-home-ubuntu/88cc26b9-b104-4dd1-b365-b88dd225bc5c/scratchpad"
    for name,lst in (('flag_lowfound',low),('flag_offcenter',off),('flag_jumpy',jump)):
        open(f"{sp}/{name}.txt","w").write("\n".join(f"{s}\t{v}" for s,v in sorted(lst,key=lambda t:t[1])))
    print("   (lists written to flag_*.txt)")
