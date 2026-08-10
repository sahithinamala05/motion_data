"""Dark + blur quality filter for the multi-dealer 10s clips (improves on the stage-2 face filter).
Gaps addressed:
  (1) explicit DARK/black-frame removal via mean grayscale intensity (was only caught indirectly);
  (2) blur judged by PROPORTION of non-blurry frames across evenly-sampled frames, not an early-exit
      at the first 30 sharp frames.
Per clip we sample every SAMPLE_STRIDE-th frame and record: median brightness/laplacian, frac_dark,
frac_blurry, frac_good (=non-dark AND non-blurry), and pass = frac_good >= PASS_FRAC.
Output: one jsonl line per clip -> merged to multi_correction/videos/quality_multi.parquet.
"""
import os, json, argparse
import numpy as np, cv2, pandas as pd

VC   = "/home/ubuntu/us-west-3-fs/live_dealer_blackjack/video_cut/batch_01"
PARQ = "/home/ubuntu/us-west-3-fs/sahithi/db/db_batch_01.parquet"
OUT  = "/home/ubuntu/us-west-3-fs/sahithi/multi_correction/videos"
PARTS= os.path.join(OUT, "parts")

DARK_THRESH  = 30      # frame is DARK if mean grayscale intensity < this (pause/black frames ~7; normal ~44-51)
BLUR_LAP     = 100     # frame is non-blurry if cv2.Laplacian(gray).var() >= this  (matches stage-2 filter)
SAMPLE_STRIDE= 5       # sample every 5th frame (~60 of 300) -> even coverage of the clip
PASS_FRAC    = 0.5     # clip passes if >= this fraction of sampled frames are non-dark AND non-blurry


def analyse(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"error": "cannot_open"}
    lums, laps = [], []; i = 0
    while True:
        if not cap.grab(): break
        if i % SAMPLE_STRIDE == 0:
            ok, f = cap.retrieve()
            if ok and f is not None:
                g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                lums.append(float(g.mean()))
                laps.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
        i += 1
    cap.release()
    n = len(lums)
    if n == 0:
        return {"error": "no_frames"}
    lums = np.array(lums); laps = np.array(laps)
    dark    = lums < DARK_THRESH
    blurry  = laps < BLUR_LAP
    good    = (~dark) & (~blurry)
    return {"n_sampled": n,
            "median_brightness": round(float(np.median(lums)), 1),
            "median_laplacian":  round(float(np.median(laps)), 1),
            "frac_dark":    round(float(dark.mean()), 4),
            "frac_blurry":  round(float(blurry.mean()), 4),
            "frac_good":    round(float(good.mean()), 4),
            "n_nonblurry":  int((~blurry).sum()),
            "pass":         bool(good.mean() >= PASS_FRAC)}


def multi_stems():
    df = pd.read_parquet(PARQ, columns=['base_name', 'single_person'])
    return list(df[df['single_person'] == False]['base_name'])   # base_name includes .mp4


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--merge', action='store_true')
    a = ap.parse_args()
    os.makedirs(PARTS, exist_ok=True)

    if a.merge:
        rows = []
        for fn in sorted(os.listdir(PARTS)):
            if fn.endswith('.jsonl'):
                for line in open(os.path.join(PARTS, fn)):
                    line = line.strip()
                    if line: rows.append(json.loads(line))
        df = pd.DataFrame(rows).drop_duplicates('base_name', keep='last')
        # fold in the parquet's existing filter columns so the manifest is one complete table
        pq = pd.read_parquet(PARQ, columns=['base_name', 'ocr_passed', 'face_passed', 'single_person'])
        df = df.merge(pq, on='base_name', how='left')
        # 'usable' = my dark+blur pass AND the existing OCR check (face left separate: multi clips fail it by design)
        df['usable'] = df['pass'].fillna(False) & df['ocr_passed'].fillna(False)
        df.to_parquet(os.path.join(OUT, "quality_multi.parquet"))
        print(f"MERGED {len(df)} clips -> {os.path.join(OUT,'quality_multi.parquet')}")
        print(f"  dark+blur pass (frac_good>=0.5): {int(df['pass'].sum())} ({100*df['pass'].mean():.1f}%)")
        print(f"  ocr_passed: {int(df['ocr_passed'].sum())} ({100*df['ocr_passed'].mean():.1f}%)")
        print(f"  USABLE (dark+blur & ocr): {int(df['usable'].sum())} ({100*df['usable'].mean():.1f}%)")
        print(f"  frac_good: mean={df['frac_good'].mean():.3f} median={df['frac_good'].median():.3f}")
        print(f"  clips with any dark frames: {(df['frac_dark']>0).sum()};  any blurry: {(df['frac_blurry']>0).sum()}")
        raise SystemExit(0)

    stems = multi_stems()
    mine = [s for i, s in enumerate(stems) if i % a.nshards == a.shard]
    part = os.path.join(PARTS, f"part_{a.shard}.jsonl")
    done = set()                                            # resume from ALL part files (robust to restarts/renames)
    import glob as _glob
    for pf in _glob.glob(os.path.join(PARTS, "part_*.jsonl")):
        for line in open(pf):
            try: done.add(json.loads(line)['base_name'])
            except Exception: pass
    fo = open(part, 'a')
    ok = 0
    for k, bn in enumerate(mine):
        if bn in done: continue
        p = os.path.join(VC, bn)
        rec = {"base_name": bn}
        rec.update(analyse(p) if os.path.exists(p) else {"error": "missing_mp4"})
        fo.write(json.dumps(rec) + "\n"); fo.flush(); ok += 1
        if (k + 1) % 200 == 0:
            print(f"shard{a.shard} {k+1}/{len(mine)} written={ok}", flush=True)
    fo.close()
    print(f"DONE shard{a.shard}: {ok} clips", flush=True)
