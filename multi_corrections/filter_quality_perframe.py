"""Per-FRAME dark+blur quality analysis (extends filter_quality.py).

Where filter_quality.py samples every 5th frame and emits one pass/fail per clip,
this analyses EVERY frame and additionally reports the exact intervals that are
dark / blurry / good — so a clip can be trimmed at frame ranges instead of dropped.

Per frame:  dark   = mean gray intensity < DARK_THRESH (black / pause screens)
            blurry = cv2.Laplacian(gray).var() < BLUR_LAP
Contiguous flagged frames collapse into [start_f, end_f, start_s, end_s] intervals (@FPS),
combined as `problematic_segments` (dark OR blurry). Per clip: frac_dark/frac_blurry/frac_good.
NOTHING is removed — every clip is stored with its problematic ranges; downstream decides
per action-window (a chunk must be clean throughout the annotated action). `pass` is kept
informational only.
OCR is NOT handled here — it's a separate project-specific phrase filter (Yifan/Dassie),
joined later; we do not gate on the DB `ocr_passed` column.

Scope: ALL remaining batch01 clips = 189K DB minus project21 `video_cut` = 94,004.
NO face-based pre-filter (face_passed is over-strict / rejects good occluded clips), so we
keep everything. Run: `launchers/run_perframe.sh` (24 shards) or
`python filter_quality_perframe.py --shard i --nshards N`, then `--merge`.
"""
import os, json, argparse, glob
import numpy as np, cv2, pandas as pd
cv2.setNumThreads(1)   # 1 OpenCV thread/proc: shards are the parallelism; avoids thread thrash

VC    = "/home/ubuntu/us-west-3-fs/live_dealer_blackjack/video_cut/batch_01"        # all source mp4s
DB    = "/home/ubuntu/us-west-3-fs/sahithi/db/db_batch_01.parquet"
P21   = "/home/ubuntu/us-west-3-fs/live_dealer_blackjack/project21_snapshot_12032025_packed/video_cut"  # 95K packed (excluded)
OUT   = "/home/ubuntu/us-west-3-fs/sahithi/rem_95k_filtered"                        # deliverable dir
PARTS = os.path.join(OUT, "parts_perframe")

DARK_THRESH = 30      # frame is DARK if mean grayscale < this
BLUR_LAP    = 150     # frame is non-blurry if Laplacian var >= this (>=150 clear, <150 blurry)
FPS         = 30.0
PASS_FRAC   = 0.5     # clip passes if >= this fraction of frames are non-dark AND non-blurry


def intervals(flags):
    """Contiguous True runs -> [(start, end)] inclusive frame indices."""
    out = []; s = None
    for i, v in enumerate(flags):
        if v and s is None: s = i
        elif not v and s is not None: out.append((s, i - 1)); s = None
    if s is not None: out.append((s, len(flags) - 1))
    return out


def with_secs(iv):
    """[(f0,f1)] -> [[f0, f1, sec0, sec1]] using FPS (sec1 is exclusive end)."""
    return [[a, b, round(a / FPS, 2), round((b + 1) / FPS, 2)] for a, b in iv]


def analyse(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"error": "cannot_open"}
    lums, laps = [], []
    while True:
        ok, f = cap.read()
        if not ok: break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        lums.append(float(g.mean()))
        laps.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
    cap.release()
    n = len(lums)
    if n == 0:
        return {"error": "no_frames"}
    lums = np.array(lums); laps = np.array(laps)
    dark = lums < DARK_THRESH; blur = laps < BLUR_LAP
    good = (~dark) & (~blur); prob = dark | blur          # problematic = dark OR blurry
    return {"n_frames": n, "fps": FPS,
            "frac_dark":   round(float(dark.mean()), 4),
            "frac_blurry": round(float(blur.mean()), 4),
            "frac_good":   round(float(good.mean()), 4),
            "problematic_segments": with_secs(intervals(prob)),   # <-- the [start_f,end_f,start_s,end_s] ranges to store
            "dark_segments":   with_secs(intervals(dark)),
            "blurry_segments": with_secs(intervals(blur)),
            "pass": bool(good.mean() >= PASS_FRAC),               # informational only; NOT a removal decision
            # raw per-frame values so ANY future threshold is an instant --rethreshold (no re-decode):
            "lum": np.rint(lums).astype(int).tolist(),
            "lap": np.rint(laps).astype(int).tolist()}


def remaining_stems():
    """ALL remaining batch01 clips = 189K DB minus project21 video_cut = 94,004.
    NO face-based pre-filter: `face_passed` layers in the over-strict face-similarity
    check, which wrongly rejects good occluded/"dealer-changed" clips (~8.7K single-person
    similarity-only rejects that likely have correct faces). We keep everything and let the
    per-frame dark/blur (+ Yifan's OCR later) decide — nothing is dropped on face flags."""
    allbn = set(pd.read_parquet(DB, columns=['base_name'])['base_name'])
    return sorted(allbn - set(os.listdir(P21)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--merge', action='store_true')
    ap.add_argument('--rethreshold', action='store_true',
                    help="re-derive segments at new --blur/--dark from stored lum/lap (INSTANT, no re-decode)")
    ap.add_argument('--blur', type=float, default=BLUR_LAP)
    ap.add_argument('--dark', type=float, default=DARK_THRESH)
    a = ap.parse_args()
    os.makedirs(PARTS, exist_ok=True)

    if a.rethreshold:
        pq = os.path.join(OUT, "quality_perframe.parquet")
        df = pd.read_parquet(pq)
        def redo(row):
            if row.get('lum') is None or len(row['lum']) == 0:
                return pd.Series({'frac_dark': None, 'frac_blurry': None, 'frac_good': None,
                                  'problematic_segments': [], 'dark_segments': [], 'blurry_segments': [], 'pass': None})
            lum = np.asarray(row['lum']); lap = np.asarray(row['lap'])
            dark = lum < a.dark; blur = lap < a.blur; good = (~dark) & (~blur); prob = dark | blur
            return pd.Series({'frac_dark': round(float(dark.mean()), 4),
                              'frac_blurry': round(float(blur.mean()), 4),
                              'frac_good': round(float(good.mean()), 4),
                              'problematic_segments': with_secs(intervals(prob)),
                              'dark_segments': with_secs(intervals(dark)),
                              'blurry_segments': with_secs(intervals(blur)),
                              'pass': bool(good.mean() >= PASS_FRAC)})
        df[['frac_dark','frac_blurry','frac_good','problematic_segments','dark_segments','blurry_segments','pass']] = df.apply(redo, axis=1)
        df.to_parquet(pq)
        anyp = df['problematic_segments'].apply(lambda s: hasattr(s,'__len__') and len(s) > 0)
        print(f"RE-THRESHOLDED (blur<{a.blur:.0f}, dark<{a.dark:.0f}) {len(df)} clips -> {pq}")
        print(f"  fully-clean {int((~anyp).sum())}  with-problematic {int(anyp.sum())}  any-blurry {int((df['frac_blurry'].fillna(0)>0).sum())}")
        raise SystemExit(0)

    if a.merge:
        rows = []
        for fn in sorted(glob.glob(PARTS + "/part_*.jsonl")):
            for line in open(fn):
                line = line.strip()
                if line: rows.append(json.loads(line))
        df = pd.DataFrame(rows).drop_duplicates('base_name', keep='last')
        db = pd.read_parquet(DB, columns=['base_name', 'ocr_passed', 'face_passed', 'single_person'])
        df = df.merge(db, on='base_name', how='left')
        # NO removal decision: store EVERY clip with its problematic frame ranges.
        # Carry DB single_person/face_passed as passthrough columns. OCR is Yifan/Dassie's
        # separate phrase-specific run (joined later) — deliberately NOT gated here.
        df.to_parquet(os.path.join(OUT, "quality_perframe.parquet"))
        any_prob = df['problematic_segments'].apply(lambda s: isinstance(s, list) and len(s) > 0)
        nerr = int(df['pass'].isna().sum())
        with open(os.path.join(OUT, "quality_summary.txt"), "w") as f:
            f.write("Per-frame dark+blur on the ~80K pre-filtered remaining batch01 clips.\n")
            f.write("Every clip is STORED with its problematic frame ranges (problematic_segments =\n")
            f.write("[start_f, end_f, start_s, end_s]); NOTHING is removed. Downstream decides per\n")
            f.write("action-window whether problematic ranges overlap the annotated action.\n")
            f.write("OCR handled separately (Yifan/Dassie phrase-specific run), joined later.\n\n")
            f.write(f"clips analysed        = {len(df)}\n")
            f.write(f"  multi / single      = {int((df['single_person']==False).sum())} / {int((df['single_person']==True).sum())}\n")
            f.write(f"errors (unreadable)   = {nerr}\n")
            f.write(f"fully clean (no problematic frames) = {int((~any_prob).sum())}\n")
            f.write(f"with problematic frames             = {int(any_prob.sum())}\n")
            f.write(f"  any dark   = {int((df['frac_dark'].fillna(0)>0).sum())}\n")
            f.write(f"  any blurry = {int((df['frac_blurry'].fillna(0)>0).sum())}\n")
        print(f"MERGED {len(df)} clips -> {OUT}/quality_perframe.parquet")
        print(f"  fully-clean {int((~any_prob).sum())}  with-problematic {int(any_prob.sum())}  errors {nerr}")
        raise SystemExit(0)

    stems = remaining_stems()
    mine = [s for i, s in enumerate(stems) if i % a.nshards == a.shard]
    part = os.path.join(PARTS, f"part_{a.shard}.jsonl")
    done = set()
    if os.path.exists(part):
        for line in open(part):
            try: done.add(json.loads(line)['base_name'])
            except Exception: pass
    fo = open(part, 'a'); ok = 0
    for k, bn in enumerate(mine):
        if bn in done: continue
        p = os.path.join(VC, bn)
        rec = {"base_name": bn}
        rec.update(analyse(p) if os.path.exists(p) else {"error": "missing_mp4"})
        fo.write(json.dumps(rec) + "\n"); fo.flush(); ok += 1
        if (k + 1) % 200 == 0:
            print(f"shard{a.shard} {k+1}/{len(mine)} ok={ok}", flush=True)
    fo.close()
    print(f"DONE shard{a.shard}: {ok} clips", flush=True)
