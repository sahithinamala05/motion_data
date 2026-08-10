"""Apply the per-frame quality result to the single-dealer pkls.

For every clip's `full_run/pkl/<cid>_dealer.pkl`, null the DWpose+WiLoR pose on the
frames flagged dark/blur (from `rem_95k_filtered/quality_perframe.parquet`'s
`problematic_segments`), keeping the good frames, and write the result to
`full_run/pkl_updated/<cid>_dealer.pkl`. A problematic frame becomes
`{'pose': None, 'removed_dark_blur': True, 'frame_dimensions': ...}`; good frames are
copied unchanged. FAST — pure pkl edits, no video decode.

Frame alignment: the pkl is a list of per-video-frame dicts (index i == video frame i),
and the quality segments are in the same video-frame index space, so `problematic_segments`
map directly onto pkl indices (guarded by len).

Run all (sharded):   launchers/run_apply_pkl.sh
Or per shard:        python apply_quality_to_pkl.py --shard i --nshards N
Comparison viz:      python apply_quality_to_pkl.py --viz --nviz 20   (left OLD | right NEW, sample across dates)

NOTE: run this only AFTER quality_perframe.parquet is finalised (the blur=... run has merged).
"""
import os, sys, json, glob, pickle, argparse, subprocess
import numpy as np, cv2, pandas as pd
cv2.setNumThreads(1)

PKL  = "/home/ubuntu/us-west-3-fs/sahithi/multi_correction/full_run/pkl"
OUT  = "/home/ubuntu/us-west-3-fs/sahithi/multi_correction/full_run/pkl_updated"
VIS  = os.path.join(OUT, "vis")
PARQ = "/home/ubuntu/us-west-3-fs/sahithi/rem_95k_filtered/quality_perframe.parquet"
VC   = "/home/ubuntu/us-west-3-fs/live_dealer_blackjack/video_cut/batch_01"
FPS  = 30.0
SUFFIX = "_dealer.pkl"

def problematic_map():
    """clip_id -> set(problematic frame indices), from quality_perframe.parquet.
    Reads only base_name + problematic_segments (NOT the big lum/lap columns)."""
    df = pd.read_parquet(PARQ, columns=['base_name', 'problematic_segments'])
    m = {}
    for bn, segs in zip(df['base_name'], df['problematic_segments']):
        cid = bn[:-4] if bn.endswith('.mp4') else bn
        s = set()
        if segs is not None:
            for seg in segs:
                for f in range(int(seg[0]), int(seg[1]) + 1):
                    s.add(f)
        m[cid] = s
    return m

def apply_one(path, prob):
    d = pickle.load(open(path, 'rb'))
    nulled = 0
    for i in prob:
        if i < len(d):
            fdim = d[i].get('frame_dimensions')
            d[i] = {'pose': None, 'removed_dark_blur': True, 'frame_dimensions': fdim}
            nulled += 1
    return d, nulled

# ── viz (left OLD | right NEW), same style as full_run/vis ──────────────────────
HAND_E = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],
          [0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20]]
LIMBS  = [[2,3],[2,6],[3,4],[4,5],[6,7],[7,8],[2,9],[9,10],[10,11],[2,12],[12,13],[13,14],[2,1],[1,15],[15,17],[1,16],[16,18]]

def draw_skeleton(ov, pose, W, H):
    if pose is None: return
    b = pose.get('bodies')
    if b is not None:
        cand = np.asarray(b['candidate']); sub = np.asarray(b['subset'])
        for pn in range(sub.shape[0]):
            for (u, v) in LIMBS:
                ia, ib = int(sub[pn, u-1]), int(sub[pn, v-1])
                if ia < 0 or ib < 0: continue
                pa = cand[ia]*[W, H]; pb = cand[ib]*[W, H]
                cv2.line(ov, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), (0,255,0), 2)
    hh = pose.get('hands')
    if hh is not None and len(np.asarray(hh)):
        hh = np.asarray(hh)
        for j in range(hh.shape[0]):
            pts = hh[j]*[W, H]
            for e in HAND_E:
                cv2.line(ov, tuple(pts[e[0]].astype(int)), tuple(pts[e[1]].astype(int)), (0,255,255), 2)

def render_compare(cid, old_pkl, new_pkl, prob):
    os.makedirs(VIS, exist_ok=True)
    cap = cv2.VideoCapture(os.path.join(VC, cid + ".mp4"))
    w = int(cap.get(3)) or 1920; h = int(cap.get(4)) or 1080
    ff = subprocess.Popen(["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pix_fmt","bgr24",
        "-s",f"{2*w}x{h}","-r",str(int(FPS)),"-i","pipe:0","-c:v","libx264","-pix_fmt","yuv420p",
        "-crf","20","-movflags","+faststart", os.path.join(VIS, cid + "_mid.mp4")], stdin=subprocess.PIPE)
    idx = 0
    while True:
        ok, img = cap.read()
        if not ok: break
        if (img.shape[1], img.shape[0]) != (w, h): img = cv2.resize(img, (w, h))
        removed = idx in prob
        left = img.copy();  draw_skeleton(left,  old_pkl[idx]['pose'] if idx < len(old_pkl) else None, w, h)
        right = img.copy(); draw_skeleton(right, new_pkl[idx]['pose'] if idx < len(new_pkl) else None, w, h)
        cv2.putText(left,  f"f{idx} {idx/FPS:4.1f}s  OLD pkl (original)", (20,44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        col = (0,0,255) if removed else (0,255,0)
        cv2.putText(right, f"NEW pkl  {'REMOVED (dark/blur)' if removed else 'KEPT'}", (20,44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)
        ff.stdin.write(np.ascontiguousarray(np.hstack([left, right])).tobytes()); idx += 1
    cap.release(); ff.stdin.close(); ff.wait()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--viz', action='store_true', help="render comparison viz for a sample instead of applying")
    ap.add_argument('--nviz', type=int, default=20)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    assert os.path.exists(PARQ), f"quality parquet not found yet: {PARQ} (run after the blur check finishes)"
    pmap = problematic_map()
    names = sorted(n for n in os.listdir(PKL) if n.endswith(SUFFIX))

    if a.viz:
        # sample clips that HAVE problematic frames, spread across dates
        cand = [(n, n[:-len(SUFFIX)]) for n in names]
        cand = [(n, c) for n, c in cand if pmap.get(c)]
        by_date = {}
        for n, c in cand: by_date.setdefault(c.split('_')[0], []).append((n, c))
        dates = sorted(by_date); pick = []; i = 0
        while len(pick) < a.nviz and any(by_date.values()):
            d = dates[i % len(dates)]
            if by_date[d]: pick.append(by_date[d].pop(0))
            i += 1
        os.makedirs(VIS, exist_ok=True)
        for n, c in pick:
            prob = pmap.get(c, set())
            old = pickle.load(open(os.path.join(PKL, n), 'rb'))
            new, _ = apply_one(os.path.join(PKL, n), prob)
            render_compare(c, old, new, prob)
            print(f"viz {c}: problematic={len(prob)}", flush=True)
        print(f"DONE viz {len(pick)} -> {VIS}", flush=True)
        raise SystemExit(0)

    # apply to a shard of all pkls (resumable: skip already-written)
    mine = names[a.shard::a.nshards]
    done = miss = 0
    for n in mine:
        outp = os.path.join(OUT, n)
        if os.path.exists(outp):
            done += 1; continue
        cid = n[:-len(SUFFIX)]
        prob = pmap.get(cid)
        if prob is None:
            miss += 1; prob = set()            # clip absent from quality parquet -> copy unchanged
        d, nulled = apply_one(os.path.join(PKL, n), prob)
        pickle.dump(d, open(outp, 'wb'))
        done += 1
        if done % 500 == 0:
            print(f"shard{a.shard} {done}/{len(mine)}", flush=True)
    print(f"DONE shard{a.shard}: wrote {done} (no-quality {miss})", flush=True)
