import sys, os, numpy as np, cv2
sys.path.insert(0,"/home/ubuntu/us-west-3-fs/sahithi/_entire_code/multi_corrections")
import dealer_faceid as m
OUTDIR="/home/ubuntu/us-west-3-fs/sahithi/multi_correction/full_run/viis_nohand_frame"
os.makedirs(OUTDIR, exist_ok=True)
SP="/tmp/claude-1000/-home-ubuntu/88cc26b9-b104-4dd1-b365-b88dd225bc5c/scratchpad"
CATS=['genuine_nohands','far_ok','noarm_nohandnear','suspect_arm','noarm_handnear']
HAND_E=[[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20]]
LIMBS=[[2,3],[2,6],[3,4],[4,5],[6,7],[7,8],[2,9],[9,10],[10,11],[2,12],[12,13],[13,14],[2,1],[1,15],[15,17],[1,16],[16,18]]

def render(stem, fi, cat):
    d=m.open_clip(stem); ref=m.front_table_scale(d); anchor=m.dealer_anchor(d,ref)
    fr=d[fi]; pose=fr['pose']; b=pose['bodies']; H,W=fr['frame_dimensions']
    cap=cv2.VideoCapture(f"{m.VC}/{stem}.mp4"); cap.set(cv2.CAP_PROP_POS_FRAMES,fi); ok,img=cap.read(); cap.release()
    if not ok: return False
    if (img.shape[1],img.shape[0])!=(W,H): img=cv2.resize(img,(W,H))
    n=None; keep_idx=[]
    if b is not None:
        cand=np.asarray(b['candidate']); sub=np.asarray(b['subset'])
        n=m.dealer_body(cand,sub,pose.get('faces'),anchor,W,H,ref)
        if n is not None and (m.valid_face_kpts(pose.get('faces'),n)<m.FACE_KPT_MIN or int((sub[n]>=0).sum())<m.MIN_BODY_KPTS): n=None
        keep=m.link_hands(cand,sub,np.asarray(pose['hands']) if pose.get('hands') is not None else None,n,W,H)
        keep_idx=[j for j,_ in keep]
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
        hh=np.asarray(pose['hands']) if pose.get('hands') is not None else np.zeros((0,21,2))
        for j in range(hh.shape[0]):
            col=(0,255,255) if j in keep_idx else (150,150,150)   # kept=yellow, dropped/other=grey
            pts=hh[j]*[W,H]
            for e in HAND_E: cv2.line(ov,tuple(pts[e[0]].astype(int)),tuple(pts[e[1]].astype(int)),col,2)
    cv2.circle(ov,(int(anchor[0]),int(anchor[1])),10,(255,0,255),2)
    cv2.putText(ov,f"{cat}  f{fi}  dealer-hands={len(keep_idx)}",(20,44),cv2.FONT_HERSHEY_SIMPLEX,1.1,(0,255,255),2)
    out=np.hstack([img,ov])
    cv2.imwrite(f"{OUTDIR}/{cat}__{stem}__f{fi}.png", cv2.resize(out,(out.shape[1]//2,out.shape[0]//2)))
    return True

tot=0
for cat in CATS:
    p=f"{SP}/nh_{cat}.txt"
    if not os.path.exists(p): continue
    for line in open(p):
        line=line.strip()
        if not line: continue
        stem,fi=line.split('\t'); fi=int(fi)
        try:
            if render(stem,fi,cat): tot+=1
        except Exception as e:
            print("ERR",cat,stem,e)
print(f"rendered {tot} frames -> {OUTDIR}")
print("per category:", {c: len([f for f in os.listdir(OUTDIR) if f.startswith(c+'__')]) for c in CATS})
