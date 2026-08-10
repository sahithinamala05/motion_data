"""Isolate ONLY the dealer in a multi-person WiLoR clip by FACE-EMBEDDING match — no fallback.

Reference = 70k_multi/<dealer_id>.png (one frontal, center-cropped frame per dealer id).
Identity resolution (no geometry fallback):
  - if the clip has a local_dealer_id in the parquet -> reference = that id's embedding.
  - else (NaN id) -> SELF-IDENTIFY: match the clip's centre face against the whole gallery
    over a few frames, majority-vote the dealer id -> reference embedding.
Then per frame: embed every detected face, cosine-match to the reference, the best match
(>=SIM_THRESH) is the dealer -> keep ONLY that DWPose body + its WiLoR hands + face.
Frames with no match are emitted empty (dealer not visible) — that is not a fallback.

Optimised: gallery embeddings cached to disk; insightface loads detection+recognition only;
frames read sequentially (no seeking); resumable; shardable.
Usage: python dealer_match.py --stem <s> [--viz] | --shard i --nshards N
"""
import os, pickle, argparse
import numpy as np, cv2, pandas as pd
from insightface.app import FaceAnalysis

VC   ="/home/ubuntu/us-west-3-fs/live_dealer_blackjack/video_cut/batch_01"
WREM ="/home/ubuntu/us-west-3-fs/sahithi/Wilor/video_cut_wilor_rem"
PARQ ="/home/ubuntu/us-west-3-fs/sahithi/db/db_batch_01.parquet"
REF  ="/home/ubuntu/us-west-3-fs/sahithi/multi_correction/70k_multi"
GAL  ="/home/ubuntu/us-west-3-fs/sahithi/multi_correction/ref_gallery_70k.npz"
OUT  ="/home/ubuntu/us-west-3-fs/sahithi/multi_correction/dealer_isolated"

SIM_THRESH   = 0.28    # cosine sim to accept a face as the dealer
SELFID_FRAMES= 12      # frames sampled to self-identify a NaN-id clip
NOSE = 0
ARMS = {'R': (2,3,4), 'L': (5,6,7)}

def norm(v): return v/(np.linalg.norm(v)+1e-9)

# ---- app (detection + recognition only -> faster) ----
_app=None
def app():
    global _app
    if _app is None:
        _app=FaceAnalysis(name='buffalo_l', allowed_modules=['detection','recognition'],
                          providers=['CUDAExecutionProvider','CPUExecutionProvider'])
        _app.prepare(ctx_id=0, det_size=(640,640))
    return _app

# ---- gallery: one embedding per dealer id, cached ----
def build_gallery():
    if os.path.exists(GAL):
        d=np.load(GAL, allow_pickle=True); return list(d['ids']), d['embs'].astype(np.float32)
    ids, embs=[], []
    files=sorted(f for f in os.listdir(REF) if f.endswith('.png'))
    for k,fn in enumerate(files):
        img=cv2.imread(os.path.join(REF,fn))
        if img is None: continue
        faces=app().get(img)
        if not faces: continue
        f=max(faces,key=lambda x:(x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
        ids.append(fn[:-4]); embs.append(norm(f.embedding))
        if (k+1)%200==0: print(f"  gallery {k+1}/{len(files)}",flush=True)
    embs=np.array(embs,dtype=np.float32)
    np.savez(GAL, ids=np.array(ids), embs=embs)
    print(f"gallery cached: {len(ids)} ids -> {GAL}",flush=True)
    return ids, embs

# ---- geometry helpers (hand linking / body pick) ----
def arm_endpoint(cand, row, side, W, H):
    sh,el,wr=ARMS[side]
    def pt(i): return cand[int(row[i])]*[W,H] if row[i]>=0 else None
    w,e,s=pt(wr),pt(el),pt(sh)
    if w is not None: return w
    if e is not None and s is not None: return e+(e-s)
    if e is not None: return e
    return None

def link_hands(cand, sub, hands, n, W, H):
    keep=[]; eps=[]
    for m in range(sub.shape[0]):
        for side in ('L','R'):
            ep=arm_endpoint(cand,sub[m],side,W,H)
            if ep is not None: eps.append((m,side,ep))
    if n is None or not eps or hands is None: return keep
    for j in range(hands.shape[0]):
        wr=hands[j][0]*[W,H]
        pidx,pside,_=min(eps,key=lambda t:np.hypot(*(wr-t[2])))
        if pidx==n: keep.append((j,pside))
    return keep

def body_for_face(bbox, cand, sub, W, H):
    cx,cy=(bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2
    best,bd=None,1e18
    for n in range(sub.shape[0]):
        ni=int(sub[n,NOSE])
        if ni<0: continue
        nx,ny=cand[ni]*[W,H]
        dd=np.hypot(nx-cx,ny-cy)
        if dd<bd: bd,best=dd,n
    return best

def compact_body(cand, sub, n):
    row=sub[n]; new_cand=[]; new_row=np.full(18,-1.0)
    for k in range(18):
        idx=int(row[k])
        if idx>=0: new_row[k]=len(new_cand); new_cand.append(cand[idx])
    nc=np.asarray(new_cand,dtype=np.float64).reshape(len(new_cand),2)
    return nc, new_row.reshape(1,18)

def center_face(faces, W):
    best,bs=None,-1
    for f in faces:
        a=(f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])
        cx=(f.bbox[0]+f.bbox[2])/2
        s=a*max(1-abs(cx/W-0.5)/0.5,0.05)
        if s>bs: bs,best=s,f
    return best

# ---- identity resolution (no geometry fallback for body selection) ----
def resolve_ref(stem, d, cap, id_map, gids, gembs):
    """Return (ref_emb, inferred_id, mode). Parquet id if present, else self-identify by face."""
    did=id_map.get(f"{stem}.mp4")
    if isinstance(did,str) and did in gids:
        return gembs[gids.index(did)], did, 'parquet'
    # self-identify: vote centre-face match across sampled frames
    N=len(d); votes={}
    for fi in np.linspace(0,max(N-1,0),SELFID_FRAMES,dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(fi)); ok,img=cap.read()
        if not ok: continue
        faces=app().get(img)
        if not faces: continue
        cf=center_face(faces,img.shape[1])
        sims=gembs@norm(cf.embedding)
        j=int(np.argmax(sims))
        if sims[j]>=SIM_THRESH: votes[j]=votes.get(j,0)+float(sims[j])
    cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    if not votes: return None, None, 'unresolved'
    j=max(votes,key=votes.get)
    return gembs[j], gids[j], 'selfid'

def open_clip(stem):
    p=os.path.join(WREM,f"{stem}_dwpose.pkl")
    if not os.path.exists(p): raise FileNotFoundError(p)
    return pickle.load(open(p,'rb'))

def process_clip(stem, id_map, gids, gembs, viz=False):
    d=open_clip(stem); cap=cv2.VideoCapture(os.path.join(VC,f"{stem}.mp4"))
    os.makedirs(OUT,exist_ok=True)
    ref, rid, mode = resolve_ref(stem,d,cap,id_map,gids,gembs)
    out=[]; picked=0
    vw=None
    if viz:
        import imageio
        vw=imageio.get_writer(os.path.join(OUT,f"{stem}_match.mp4"),fps=15,codec='libx264',
                              macro_block_size=None,ffmpeg_params=['-pix_fmt','yuv420p'])
    HAND_E=[[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20]]
    LIMBS=[[2,3],[2,6],[3,4],[4,5],[6,7],[7,8],[2,9],[9,10],[10,11],[2,12],[12,13],[13,14],[2,1],[1,15],[15,17],[1,16],[16,18]]
    for fr in d:
        ok,img=cap.read()
        if not ok: break
        H,W=fr['frame_dimensions']
        if (img.shape[1],img.shape[0])!=(W,H): img=cv2.resize(img,(W,H))
        pose=fr['pose']; b=pose['bodies']
        n=None; bbox=None; keep_idx=[]
        if b is not None and ref is not None:
            cand=np.asarray(b['candidate']); sub=np.asarray(b['subset'])
            faces=app().get(img)
            best,bs=None,-1
            for f in faces:
                s=float(gembs_dot(f,ref))
                if s>bs: bs,best=s,f
            if best is not None and bs>=SIM_THRESH:
                bbox=best.bbox
                n=body_for_face(bbox,cand,sub,W,H)
                hands=np.asarray(pose['hands']) if pose.get('hands') is not None else None
                keep=link_hands(cand,sub,hands,n,W,H); keep_idx=[j for j,_ in keep]
        if n is not None:
            picked+=1
            cand=np.asarray(b['candidate']); sub=np.asarray(b['subset'])
            hands=np.asarray(pose['hands']); isr=pose.get('hands_is_right',[])
            faces_kp=np.asarray(pose['faces']) if pose.get('faces') is not None else None
            dc,ds=compact_body(cand,sub,n)
            out_fr={'pose':{'bodies':{'candidate':dc,'subset':ds},
                            'hands':hands[keep_idx] if keep_idx else hands[:0],
                            'hands_is_right':[isr[j] for j in keep_idx] if isr else [],
                            'faces':faces_kp[n:n+1] if faces_kp is not None and n<len(faces_kp) else faces_kp},
                    'frame_dimensions':fr['frame_dimensions']}
        else:
            out_fr={'pose':{'bodies':None,'hands':np.zeros((0,21,2)),'hands_is_right':[],'faces':None},
                    'frame_dimensions':fr['frame_dimensions'],'dealer_not_found':True}
        out.append(out_fr)
        if vw is not None:
            ov=img.copy()
            if b is not None:
                cand=np.asarray(b['candidate']); sub=np.asarray(b['subset'])
                for pn in range(sub.shape[0]):
                    col=(0,255,0) if pn==n else (0,0,255)
                    for (u,v) in LIMBS:
                        ia,ib=int(sub[pn,u-1]),int(sub[pn,v-1])
                        if ia<0 or ib<0: continue
                        pa=cand[ia]*[W,H]; pb=cand[ib]*[W,H]
                        cv2.line(ov,(int(pa[0]),int(pa[1])),(int(pb[0]),int(pb[1])),col,2)
                hh=np.asarray(pose['hands'])
                for j in range(hh.shape[0]):
                    col=(0,255,255) if j in keep_idx else (128,128,128)
                    pts=hh[j]*[W,H]
                    for e in HAND_E: cv2.line(ov,tuple(pts[e[0]].astype(int)),tuple(pts[e[1]].astype(int)),col,2)
            if bbox is not None: cv2.rectangle(ov,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,255,0),2)
            cv2.putText(ov,f"dealer={'FOUND' if n is not None else 'none'} id={rid}({mode})",(20,40),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0) if n is not None else (0,0,255),2)
            vw.append_data(np.hstack([img,ov])[:,:,::-1])
    cap.release()
    if vw is not None: vw.close()
    op=os.path.join(OUT,f"{stem}_dealer.pkl"); pickle.dump(out,open(op,'wb'))
    return len(out),picked,rid,mode,op

def gembs_dot(f, ref):
    return np.dot(norm(f.embedding), ref)

def id_map_from_parquet():
    df=pd.read_parquet(PARQ,columns=['base_name','local_dealer_id'])
    return dict(zip(df['base_name'], df['local_dealer_id']))

def multi_stems():
    df=pd.read_parquet(PARQ,columns=['base_name','single_person'])
    return [b[:-4] for b in df[df['single_person']==False]['base_name']]

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--stem'); ap.add_argument('--viz',action='store_true')
    ap.add_argument('--shard',type=int,default=0); ap.add_argument('--nshards',type=int,default=1)
    ap.add_argument('--limit',type=int,default=0)
    a=ap.parse_args()
    gids,gembs=build_gallery()
    id_map=id_map_from_parquet()
    if a.stem:
        n,pk,rid,mode,op=process_clip(a.stem,id_map,gids,gembs,viz=a.viz)
        print(f"DONE {a.stem}: id={rid}({mode}) dealer in {pk}/{n} ({100*pk/max(n,1):.1f}%) -> {op}",flush=True)
    else:
        stems=multi_stems()
        if a.nshards>1: stems=[s for i,s in enumerate(stems) if i%a.nshards==a.shard]
        if a.limit: stems=stems[:a.limit]
        ok=unres=0
        for k,s in enumerate(stems):
            if os.path.exists(os.path.join(OUT,f"{s}_dealer.pkl")): continue
            if not os.path.exists(os.path.join(WREM,f"{s}_dwpose.pkl")): continue
            try:
                n,pk,rid,mode,op=process_clip(s,id_map,gids,gembs)
                ok+=1
                if mode=='unresolved': unres+=1
            except Exception as e:
                print(f"ERR {s}: {e}",flush=True)
            if (k+1)%50==0: print(f"{k+1}/{len(stems)} ok={ok} unresolved={unres}",flush=True)
        print(f"DONE shard {a.shard}: ok={ok} unresolved={unres}",flush=True)
