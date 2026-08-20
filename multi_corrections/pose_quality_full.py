"""Full-set pose-quality scorer — reconstructed verbatim from the original prototype.
clip_metrics is byte-identical to the script that produced pose_quality_sample.csv.

Modes:
  validate  : recompute metrics for the CSV's sample cids from full_run/pkl, diff vs CSV.
  full      : compute metrics for ALL _dealer.pkl in --pkl, save features + bad, print counts.

bad = z(att_p90)+z(bonecov)+z(jitter_p95) - z(frac_body) - z(med_valid_body)
"""
import os, sys, pickle, argparse, csv
import numpy as np, pandas as pd
from multiprocessing import Pool

RSHO,RELB,RWRI,LSHO,LELB,LWRI = 2,3,4,5,6,7
PKL = None  # set in main

def clip_metrics(cid):
    try:
        d = pickle.load(open(f"{PKL}/{cid}_dealer.pkl", 'rb'))
    except Exception:
        return None
    if not d: return None
    H, W = d[0]['frame_dimensions']
    nfr = len(d); body=0; hand2=0; face=0
    nvalid=[]; faceval=[]; att=[]
    bones={'faR':[],'faL':[],'sh':[]}; jit=[]; prev={}
    def gp(cand, sub, j):
        i=int(sub[0,j])
        if i<0 or i>=len(cand): return None
        x,y=cand[i]
        return np.array([x*W, y*H]) if (np.isfinite(x) and np.isfinite(y)) else None
    def dist(a,b): return float(np.linalg.norm(a-b)) if (a is not None and b is not None) else np.nan
    for fr in d:
        pose=fr.get('pose')
        if pose is None or pose.get('bodies') is None: continue
        cand=np.asarray(pose['bodies']['candidate']); sub=np.asarray(pose['bodies']['subset'])
        if sub.ndim!=2: continue
        body+=1; nvalid.append(int((sub[0]>=0).sum()))
        f=pose.get('faces')
        if f is not None:
            f=np.asarray(f)
            if f.size:
                v=np.isfinite(f[0]).all(1)&(f[0,:,0]>=0)&(f[0,:,0]<=1)&(f[0,:,1]>=0)&(f[0,:,1]<=1)
                faceval.append(int(v.sum())); face += v.sum()>=60
        rs,re,rw=gp(cand,sub,RSHO),gp(cand,sub,RELB),gp(cand,sub,RWRI)
        ls,le,lw=gp(cand,sub,LSHO),gp(cand,sub,LELB),gp(cand,sub,LWRI)
        faR,faL,sh=dist(re,rw),dist(le,lw),dist(rs,ls)
        bones['faR'].append(faR); bones['faL'].append(faL); bones['sh'].append(sh)
        scale = sh if np.isfinite(sh) and sh>0 else np.nanmean([faR,faL])
        hh=pose.get('hands')
        if hh is not None:
            hh=np.asarray(hh)
            if hh.ndim==3 and hh.shape[0]>0:
                hand2 += hh.shape[0]>=2
                wr=[w for w in (rw,lw) if w is not None]; fa=np.nanmean([faR,faL])
                if wr and np.isfinite(fa) and fa>0:
                    for k in range(hh.shape[0]):
                        root=hh[k,0]*[W,H]
                        att.append(min(np.linalg.norm(root-w) for w in wr)/fa)
        cur={}
        for nm,j in (('rw',RWRI),('lw',LWRI),('re',RELB),('le',LELB)):
            p=gp(cand,sub,j)
            if p is not None: cur[nm]=p
        if np.isfinite(scale) and scale>0:
            for nm,p in cur.items():
                if nm in prev: jit.append(np.linalg.norm(p-prev[nm])/scale)
        prev=cur
    def cov(x):
        x=np.array([v for v in x if np.isfinite(v)])
        return float(np.std(x)/np.median(x)) if len(x)>=5 and np.median(x)>0 else np.nan
    return dict(cid=cid, n_frames=nfr, frac_body=body/max(nfr,1), frac_2hands=hand2/max(body,1),
        med_valid_body=float(np.median(nvalid)) if nvalid else 0.0,
        med_face_valid=float(np.median(faceval)) if faceval else 0.0,
        att_p90=float(np.nanpercentile(att,90)) if att else np.nan,
        bonecov=float(np.nanmean([cov(bones['faR']),cov(bones['faL']),cov(bones['sh'])])),
        jitter_p95=float(np.nanpercentile(jit,95)) if jit else np.nan)

def zscore(x, hi_bad=True):
    x=x.astype(float); m,s=x.mean(),x.std()
    zz=(x-m)/s if s>0 else x*0
    return zz if hi_bad else -zz

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("mode",choices=["validate","full"])
    ap.add_argument("--pkl",required=True)
    ap.add_argument("--out",default="/home/ubuntu/us-west-3-fs/sahithi/rem_95k_filtered/pose_quality_newset/pose_quality_full.csv")
    ap.add_argument("--procs",type=int,default=16)
    a=ap.parse_args()
    PKL=a.pkl

    if a.mode=="validate":
        csvp="/home/ubuntu/us-west-3-fs/sahithi/rem_95k_filtered/pose_quality_newset/pose_quality_sample.csv"
        ref={r["cid"]:r for r in csv.DictReader(open(csvp))}
        cids=list(ref)[:200]
        with Pool(a.procs) as p: rows=[r for r in p.map(clip_metrics,cids,chunksize=8) if r]
        cols=["frac_body","frac_2hands","med_valid_body","med_face_valid","att_p90","bonecov","jitter_p95"]
        maxdiff={c:0.0 for c in cols}; nbad=0
        for r in rows:
            R=ref[r["cid"]]
            for c in cols:
                a_=r[c]; b_=R[c]
                if b_.strip()=="" : continue
                b_=float(b_)
                if not (np.isfinite(a_) and np.isfinite(b_)):
                    if np.isfinite(a_)!=np.isfinite(b_): nbad+=1
                    continue
                maxdiff[c]=max(maxdiff[c],abs(a_-b_))
        print(f"validated {len(rows)} clips vs CSV. nan-mismatches={nbad}")
        for c in cols: print(f"  max|Δ| {c:16s} = {maxdiff[c]:.2e}")
        ok=all(v<1e-4 for v in maxdiff.values()) and nbad==0
        print("EXTRACTOR MATCH:", "OK" if ok else "MISMATCH")
        sys.exit(0 if ok else 1)

    # full
    names=sorted(f[:-len('_dealer.pkl')] for f in os.listdir(PKL) if f.endswith('_dealer.pkl'))
    print(f"scoring {len(names):,} clips from {PKL} ...",flush=True)
    with Pool(a.procs) as p:
        rows=[r for r in p.map(clip_metrics,names,chunksize=16) if r]
    df=pd.DataFrame(rows)
    print(f"got metrics for {len(df):,} clips",flush=True)
    # population z-scores (correct full-run normalization)
    df['bad']=(zscore(df['att_p90'])+zscore(df['bonecov'])+zscore(df['jitter_p95'])
               +zscore(df['frac_body'],hi_bad=False)+zscore(df['med_valid_body'],hi_bad=False)).round(2)
    # also: sample-fixed normalization (comparable to the 2k-sample CSV's 'bad' scale)
    s=pd.read_csv("/home/ubuntu/us-west-3-fs/sahithi/rem_95k_filtered/pose_quality_newset/pose_quality_sample.csv")
    def zfix(col,hi_bad=True):
        m,sd=s[col].astype(float).mean(),s[col].astype(float).std()
        zz=(df[col].astype(float)-m)/sd if sd>0 else df[col]*0
        return zz if hi_bad else -zz
    df['bad_samplez']=(zfix('att_p90')+zfix('bonecov')+zfix('jitter_p95')
                       +zfix('frac_body',False)+zfix('med_valid_body',False)).round(2)
    df=df.sort_values('bad',ascending=False)
    df.to_csv(a.out,index=False)
    print("saved ->",a.out)
    n=len(df)
    for label,col in (("population-z","bad"),("sample-fixed-z","bad_samplez")):
        v=df[col].dropna()
        print(f"\n[{label}]  scored={len(v):,}")
        for t in (4.0,4.5,5.0):
            c=(v>=t).sum(); print(f"   bad >= {t}: {c:,} ({100*c/len(v):.2f}%)")
        print(f"   min {v.min():.2f}  max {v.max():.2f}")
    print("\nfeatures with NaN (no valid body/hands frames): %d clips"%df['att_p90'].isna().sum())
    print("DONE",flush=True)
