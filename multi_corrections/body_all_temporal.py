"""DWpose BODY temporal inconsistency over ALL 18 body keypoints (except hands).

Source: dwpose/batch_01/dwpose/<cid>_dwpose.pkl  pose.bodies (OpenPose-18 candidate/subset).
Per frame, for every VALID body joint (subset index >= 0 in both this and the previous
frame), compute ||p_t - p_{t-1}|| / shoulder_width. The frame's value = MAX over joints
(worst-moving joint) so a single teleporting joint is caught, not diluted. Also records
which joints are worst. Bad frame = value > THRESH (0.5 shoulder-widths); segment >= 2 consec.

Per clip -> OUT/<cid>.json:
  {clip, n_frames, p50, p95, max, n_bad, segments, worst_joints}
worst_joints = {joint_index: times_it_was_the_worst_in_a_bad_frame}. Sharded, resume-safe.
"""
import os, sys, json, pickle, argparse, collections
import numpy as np

DW="/home/ubuntu/us-west-3-fs/live_dealer_blackjack/dwpose/batch_01/dwpose"
VCSET="/home/ubuntu/us-west-3-fs/sahithi/Wilor/video_cut_wilor"   # same 95k clip set as the others
OUT="/home/ubuntu/us-west-3-fs/sahithi/hand_consistency/body_temporal"
FPS=30.0; THRESH=1.0; MIN_RUN=2   # bad = a joint moved more than a full shoulder-width in one frame
RSHO,LSHO=2,5
JOINT_NAMES={0:'nose',1:'neck',2:'Rsho',3:'Relb',4:'Rwri',5:'Lsho',6:'Lelb',7:'Lwri',
 8:'Rhip',9:'Rknee',10:'Rank',11:'Lhip',12:'Lknee',13:'Lank',14:'Reye',15:'Leye',16:'Rear',17:'Lear'}
# exclude below the hip (knees 9/12, ankles 10/13) — occluded behind the table, unreliable
ALLOWED={0,1,2,3,4,5,6,7,8,11,14,15,16,17}

def joints(pose):
    """return {j: np.array([x,y])} for all valid body joints (subset>=0, finite)."""
    b=pose.get('bodies') if pose else None
    if b is None: return {}
    c=np.asarray(b['candidate']); s=np.asarray(b['subset'])
    if s.ndim!=2: return {}
    out={}
    for j in range(min(18, s.shape[1])):
        if j not in ALLOWED: continue          # skip knees/ankles (below hip)
        i=int(s[0,j])
        if 0<=i<len(c):
            x,y=c[i]
            if np.isfinite(x) and np.isfinite(y): out[j]=np.array([x,y])
    return out
def segs(mask):
    out=[]; st=None
    for i,v in enumerate(list(mask)+[False]):
        if v and st is None: st=i
        elif not v and st is not None:
            if (i-st)>=MIN_RUN: out.append([st,i-1,round(st/FPS,2),round(i/FPS,2)])
            st=None
    return out

def process(cid):
    try: D=pickle.load(open(f"{DW}/{cid}_dwpose.pkl",'rb'))
    except Exception: return None
    n=len(D); vals=[]; mask=[False]*n; prev={}; worst=collections.Counter()
    for k in range(n):
        cur=joints(D[k].get('pose') if isinstance(D[k],dict) else None)
        sw=None
        if RSHO in cur and LSHO in cur: sw=np.linalg.norm(cur[RSHO]-cur[LSHO])
        val=np.nan; wj=None
        if sw and sw>0 and prev:
            per={j:np.linalg.norm(cur[j]-prev[j])/sw for j in cur if j in prev}
            if per:
                wj=max(per,key=per.get); val=float(per[wj])
        if np.isfinite(val):
            vals.append(val)
            if val>THRESH: mask[k]=True; worst[wj]+=1
        prev=cur
    def pct(q): return round(float(np.percentile(vals,q)),4) if vals else None
    rec=dict(clip=cid, n_frames=n, source='dwpose_body_allpts',
             p50=pct(50), p95=pct(95), max=round(float(max(vals)),4) if vals else None,
             n_bad=int(sum(mask)), segments=segs(mask),
             worst_joints={JOINT_NAMES[j]:int(c) for j,c in worst.most_common()})
    json.dump(rec, open(f"{OUT}/{cid}.json","w"))
    return rec

def all_ids():
    return sorted(f[:-len('_dwpose.pkl')] for f in os.listdir(VCSET) if f.endswith('_dwpose.pkl'))

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument('--shard',type=int,default=0); ap.add_argument('--nshards',type=int,default=1)
    a=ap.parse_args(); os.makedirs(OUT,exist_ok=True)
    ids=all_ids(); mine=[c for i,c in enumerate(ids) if i%a.nshards==a.shard]
    print(f"shard {a.shard}/{a.nshards}: {len(mine)} clips",flush=True)
    done=0
    for cid in mine:
        try: process(cid)      # overwrite (schema changed) — no resume-skip
        except Exception: pass
        done+=1
        if done%2000==0: print(f"  shard{a.shard}: {done}/{len(mine)}",flush=True)
    print(f"SHARD_{a.shard}_DONE",flush=True)
