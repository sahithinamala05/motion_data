"""Full-set pose-quality scorer for the single-dealer DWpose pkls.

Computes a per-clip "badness" score from DWpose geometry+temporal signals (no pose
confidence is available, so we derive quality from geometry). Reconstructed verbatim
from the original prototype, minus the two metrics that were never used in the score
(frac_2hands, med_face_valid) — dropping them does NOT change `bad` (both had 0 weight).

Metrics per clip:
  frac_body        fraction of frames with a valid body skeleton            (higher = better)
  med_valid_body   median # valid body keypoints, of 18                     (higher = better)
  att_p90          90th-pct hand-root -> nearest DWpose wrist / forearm     (higher = worse: floating/mislinked hands)
  bonecov          mean temporal coeff-of-variation of forearm+shoulder     (higher = worse: unstable/flickering limbs)
  jitter_p95       95th-pct frame-to-frame wrist/elbow disp / shoulder-width (higher = worse: teleporting keypoints)

Composite:
  bad = z(att_p90) + z(bonecov) + z(jitter_p95) - z(frac_body) - z(med_valid_body)
z-scores are taken over the population being scored. Clips whose skeleton never
resolves get NaN metrics (no usable pose) and no score -> treat as a separate reject bucket.

Modes:
  validate  : recompute metrics for a reference CSV's cids and diff (sanity check).
  full      : score ALL _dealer.pkl in --pkl, write features + bad, print threshold counts.
"""
import os, sys, pickle, argparse, csv
import numpy as np, pandas as pd
from multiprocessing import Pool

RSHO, RELB, RWRI, LSHO, LELB, LWRI = 2, 3, 4, 5, 6, 7   # OpenPose-18 body indices
PKL = None  # set in main

def clip_metrics(cid):
    try:
        d = pickle.load(open(f"{PKL}/{cid}_dealer.pkl", 'rb'))
    except Exception:
        return None
    if not d: return None
    H, W = d[0]['frame_dimensions']
    nfr = len(d); body = 0
    nvalid = []; att = []
    bones = {'faR': [], 'faL': [], 'sh': []}; jit = []; prev = {}
    def gp(cand, sub, j):
        i = int(sub[0, j])
        if i < 0 or i >= len(cand): return None
        x, y = cand[i]
        return np.array([x * W, y * H]) if (np.isfinite(x) and np.isfinite(y)) else None
    def dist(a, b): return float(np.linalg.norm(a - b)) if (a is not None and b is not None) else np.nan
    for fr in d:
        pose = fr.get('pose')
        if pose is None or pose.get('bodies') is None: continue
        cand = np.asarray(pose['bodies']['candidate']); sub = np.asarray(pose['bodies']['subset'])
        if sub.ndim != 2: continue
        body += 1; nvalid.append(int((sub[0] >= 0).sum()))
        rs, re, rw = gp(cand, sub, RSHO), gp(cand, sub, RELB), gp(cand, sub, RWRI)
        ls, le, lw = gp(cand, sub, LSHO), gp(cand, sub, LELB), gp(cand, sub, LWRI)
        faR, faL, sh = dist(re, rw), dist(le, lw), dist(rs, ls)
        bones['faR'].append(faR); bones['faL'].append(faL); bones['sh'].append(sh)
        scale = sh if np.isfinite(sh) and sh > 0 else np.nanmean([faR, faL])
        hh = pose.get('hands')
        if hh is not None:
            hh = np.asarray(hh)
            if hh.ndim == 3 and hh.shape[0] > 0:
                wr = [w for w in (rw, lw) if w is not None]; fa = np.nanmean([faR, faL])
                if wr and np.isfinite(fa) and fa > 0:
                    for k in range(hh.shape[0]):
                        root = hh[k, 0] * [W, H]
                        att.append(min(np.linalg.norm(root - w) for w in wr) / fa)
        cur = {}
        for nm, j in (('rw', RWRI), ('lw', LWRI), ('re', RELB), ('le', LELB)):
            p = gp(cand, sub, j)
            if p is not None: cur[nm] = p
        if np.isfinite(scale) and scale > 0:
            for nm, p in cur.items():
                if nm in prev: jit.append(np.linalg.norm(p - prev[nm]) / scale)
        prev = cur
    def cov(x):
        x = np.array([v for v in x if np.isfinite(v)])
        return float(np.std(x) / np.median(x)) if len(x) >= 5 and np.median(x) > 0 else np.nan
    return dict(cid=cid, n_frames=nfr, frac_body=body / max(nfr, 1),
        med_valid_body=float(np.median(nvalid)) if nvalid else 0.0,
        att_p90=float(np.nanpercentile(att, 90)) if att else np.nan,
        bonecov=float(np.nanmean([cov(bones['faR']), cov(bones['faL']), cov(bones['sh'])])),
        jitter_p95=float(np.nanpercentile(jit, 95)) if jit else np.nan)

def zscore(x, hi_bad=True):
    x = x.astype(float); m, s = x.mean(), x.std()
    zz = (x - m) / s if s > 0 else x * 0
    return zz if hi_bad else -zz

def compute_bad(df):
    return (zscore(df['att_p90']) + zscore(df['bonecov']) + zscore(df['jitter_p95'])
            + zscore(df['frac_body'], hi_bad=False) + zscore(df['med_valid_body'], hi_bad=False)).round(2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["validate", "full"])
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--out", default="/home/ubuntu/us-west-3-fs/sahithi/rem_95k_filtered/pose_quality_newset/pose_quality_full.csv")
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    PKL = a.pkl

    if a.mode == "validate":
        ref = {r["cid"]: r for r in csv.DictReader(open(a.out))}
        cids = list(ref)[:200]
        with Pool(a.procs) as p: rows = [r for r in p.map(clip_metrics, cids, chunksize=8) if r]
        cols = ["frac_body", "med_valid_body", "att_p90", "bonecov", "jitter_p95"]
        maxd = {c: 0.0 for c in cols}
        for r in rows:
            R = ref[r["cid"]]
            for c in cols:
                if c in R and R[c].strip() and np.isfinite(r[c]):
                    maxd[c] = max(maxd[c], abs(r[c] - float(R[c])))
        print(f"validated {len(rows)} clips")
        for c in cols: print(f"  max|Δ| {c:16s} = {maxd[c]:.2e}")
        sys.exit(0)

    names = sorted(f[:-len('_dealer.pkl')] for f in os.listdir(PKL) if f.endswith('_dealer.pkl'))
    print(f"scoring {len(names):,} clips from {PKL} ...", flush=True)
    with Pool(a.procs) as p:
        rows = [r for r in p.map(clip_metrics, names, chunksize=16) if r]
    df = pd.DataFrame(rows)
    df['bad'] = compute_bad(df)
    df = df.sort_values('bad', ascending=False)
    df.to_csv(a.out, index=False)
    print("saved ->", a.out)
    v = df['bad'].dropna()
    print(f"scored {len(v):,} | no-pose (NaN) {int(df['att_p90'].isna().sum()):,}")
    for t in (3.0, 4.0, 4.5, 5.0):
        c = int((v >= t).sum()); print(f"  bad >= {t}: {c:,} ({100*c/len(v):.2f}%)")
    print("DONE", flush=True)
