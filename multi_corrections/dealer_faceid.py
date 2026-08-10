"""Select ONLY the dealer from a multi-person WiLoR clip — reference-free, no fallback.

Rule (fixed camera): the dealer is the MIDDLE person at the FRONT table.
  STEP 1 - front table first: on a fixed camera the front-table dealer has the LARGE/near FACE;
    dealers at far/adjacent tables have small faces. Compute the clip's front-table face size and
    treat only bodies whose face >= NEAR_FRAC * that size as "at the front table" (face size is a
    robust depth proxy — unlike torso it doesn't need hips/wrists). Everyone else is ignored.
  STEP 2 - the dealer's seat, from DWPose only (no insightface, no video decode):
    a. per frame, take the front-table body whose head is nearest the frame middle point;
    b. median over the clip = the dealer's stable seat (ANCHOR) — a far-table body can never
       pull it, so it stays on the front dealer even if her face is briefly occluded;
    c. every frame: keep the FRONT-table body whose head sits at the anchor + its hands.

No identity gallery, no parquet, no insightface -> pure DWPose/WiLoR, applies to every clip.
Output: single-dealer pkl (same schema as input) + optional viz.
Usage:  python dealer_faceid.py --stem <clip_stem> [--viz]
"""
import os, pickle, argparse
import numpy as np, cv2, pandas as pd

VC     = "/home/ubuntu/us-west-3-fs/live_dealer_blackjack/video_cut/batch_01"
WREM   = "/home/ubuntu/us-west-3-fs/sahithi/Wilor/video_cut_wilor_rem"
PARQ   = "/home/ubuntu/us-west-3-fs/sahithi/db/db_batch_01.parquet"
OUTDIR = "/home/ubuntu/us-west-3-fs/sahithi/multi_correction/full_run"
PKL_DIR= os.path.join(OUTDIR, "pkl")
VIS_DIR= os.path.join(OUTDIR, "vis")

NOSE = 0
NECK = 1
ARMS = {'R': (2, 3, 4), 'L': (5, 6, 7)}
MAXD_FRAC     = 0.22        # a body head must be within this * frame-diagonal of the anchor to count as the dealer
FACE_KPT_MIN  = 66          # min valid DWPose face keypoints (of 68) for the dealer this frame; below this
                            # the face isn't really there (turned away / absent / DWPose ghost) -> drop the frame
MIN_BODY_KPTS = 8           # min valid body joints for the selected body; a sparse fragment (empty-table
                            # ghost) has fewer -> drop, even if DWPose hallucinated a face on it
NEAR_FRAC     = 0.6         # a body is "at the front table" if its FACE size >= this * the clip's front-table
                            # face size; smaller faces are at FAR/adjacent tables and are ignored entirely

# ---- hand linking / body compaction ----
def arm_endpoint(cand, sub_row, side, W, H):
    sh, el, wr = ARMS[side]
    def pt(i):
        return cand[int(sub_row[i])] * [W, H] if sub_row[i] >= 0 else None
    w, e, s = pt(wr), pt(el), pt(sh)
    if w is not None:
        return w
    if e is not None and s is not None:
        return e + (e - s)
    if e is not None:
        return e
    return None


OTHER_MARGIN = 0.8          # drop a dealer-candidate hand if a behind person's arm is < this * the dealer distance

def other_arm_endpoints(cand, sub, n, W, H):
    """Arm endpoints of every OTHER (behind / adjacent) detected person — used as negatives."""
    eps = []
    for m in range(sub.shape[0]):
        if m == n:
            continue
        for s in ('R', 'L'):
            ep = arm_endpoint(cand, sub[m], s, W, H)
            if ep is not None:
                eps.append(ep)
    return eps


def link_hands(cand, sub, hands, n, W, H):
    """Keep ONLY the middle dealer's hands. The dealer has two wrists (R, L); each wrist takes
    its single nearest WiLoR hand -> at most 2 hands (a person has two hands), which structurally
    excludes surrounding hands and collapses duplicate detections. The wrist point may be missing
    -> arm_endpoint falls back to forearm-extrapolation / elbow. Finally, if a candidate hand is
    much closer to a BEHIND person's arm than to the dealer's, it is that person's -> drop it."""
    if n is None or hands is None or hands.shape[0] == 0:
        return []
    targets = {s: arm_endpoint(cand, sub[n], s, W, H) for s in ('R', 'L')}
    targets = {s: ep for s, ep in targets.items() if ep is not None}   # wrist -> forearm -> elbow fallback
    if not targets:
        return []
    others = other_arm_endpoints(cand, sub, n, W, H)
    wrists = hands[:, 0, :] * [W, H]                                   # (M,2) WiLoR wrist per hand
    nearest = {s: min(range(len(wrists)), key=lambda j: np.hypot(*(wrists[j] - ep)))
               for s, ep in targets.items()}                           # side -> nearest hand idx
    if nearest.get('R') == nearest.get('L'):                           # same hand won both wrists -> closer wrist keeps it
        s = min(targets, key=lambda s: np.hypot(*(wrists[nearest[s]] - targets[s])))
        nearest = {s: nearest[s]}
    keep = []
    for s, j in nearest.items():
        dd = np.hypot(*(wrists[j] - targets[s]))                       # hand -> dealer arm
        if others and min(np.hypot(*(wrists[j] - eo)) for eo in others) < OTHER_MARGIN * dd:
            continue                                                   # belongs to a behind person -> drop
        keep.append((j, s))
    return keep


def compact_body(cand, sub, n):
    row = sub[n]
    new_cand, new_row = [], np.full(18, -1.0)
    for k in range(18):
        idx = int(row[k])
        if idx >= 0:
            new_row[k] = len(new_cand)
            new_cand.append(cand[idx])
    nc = np.asarray(new_cand, dtype=np.float64).reshape(len(new_cand), 2)
    return nc, new_row.reshape(1, 18)


# ---- STEP 1: who is at the FRONT table (near/large), vs a far/adjacent table (small) ----
def head_px(cand, sub_row, W, H):
    def pt(i):
        return cand[int(sub_row[i])] * [W, H] if sub_row[i] >= 0 else None
    h = pt(NOSE)
    if h is None: h = pt(NECK)
    return h

def face_size(faces, n, W, H):
    """A person's FACE size in px (geom-mean of the DWPose face-keypoint bbox) = a robust depth
    proxy on a fixed camera: bigger face = closer = at the front table. Unlike torso height it
    does NOT depend on hips/wrists being detected. 0 if body n has no usable face."""
    if faces is None:
        return 0.0
    faces = np.asarray(faces)
    if n >= faces.shape[0]:
        return 0.0
    f = faces[n]
    inr = ((f >= 0) & (f <= 1)).all(axis=1)
    if inr.sum() < 5:
        return 0.0
    w = np.ptp(f[inr, 0]) * W; h = np.ptp(f[inr, 1]) * H
    return float((max(w, 1.0) * max(h, 1.0)) ** 0.5)

def front_table_scale(d):
    """Clip-level front-table FACE size = high percentile of all face sizes (the dealer is present
    most of the time, so this ~= the front dealer's face size). Far-table faces sit well below it."""
    sizes = []
    for fr in d:
        b = fr['pose']['bodies']
        if b is None: continue
        faces = fr['pose'].get('faces'); sub = np.asarray(b['subset'])
        H, W = fr['frame_dimensions']
        for n in range(sub.shape[0]):
            s = face_size(faces, n, W, H)
            if s > 0: sizes.append(s)
    return float(np.percentile(sizes, 80)) if sizes else 0.0

def near_bodies(faces, sub, W, H, ref):
    """Indices of bodies at the FRONT table (face size >= NEAR_FRAC * ref)."""
    return [n for n in range(sub.shape[0]) if face_size(faces, n, W, H) >= NEAR_FRAC * ref]


# ---- STEP 2: the dealer's seat (anchor), from DWPose only — no insightface, no video decode ----
def dealer_anchor(d, ref):
    """Median, over the clip, of the FRONT-table body whose head is nearest the frame centre
    = the dealer's stable seat. Computed purely from the DWPose pkl. Far/adjacent-table bodies
    are excluded (small face), so the anchor can't drift onto them even when the front dealer's
    face is briefly occluded. Defaults to the frame middle point if no front-table body is seen."""
    H, W = d[0]['frame_dimensions']
    mid = np.array([W / 2.0, H / 2.0])
    centres = []
    for fr in d:
        b = fr['pose']['bodies']
        if b is None: continue
        cand = np.asarray(b['candidate']); sub = np.asarray(b['subset'])
        heads = [head_px(cand, sub[n], W, H) for n in near_bodies(fr['pose'].get('faces'), sub, W, H, ref)]
        heads = [h for h in heads if h is not None]
        if heads:
            centres.append(min(heads, key=lambda h: np.hypot(*(h - mid))))
    return np.median(np.array(centres), axis=0) if centres else mid

def dealer_body(cand, sub, faces, anchor, W, H, ref):
    """Index of the FRONT-table body whose head sits at the anchor (the central dealer), or None.
    Only front-table bodies are considered — an adjacent-table person can never be picked."""
    best, bd = None, 1e18
    for n in near_bodies(faces, sub, W, H, ref):
        h = head_px(cand, sub[n], W, H)
        if h is None: continue
        dd = np.hypot(*(h - anchor))
        if dd < bd: bd, best = dd, n
    if best is None or bd > MAXD_FRAC * np.hypot(W, H):
        return None
    return best


def valid_face_kpts(faces, n):
    """Count of DWPose face keypoints (of 68) for body n that are in-frame [0,1] = a real,
    visible face. Low when the dealer is turned away / absent. 0 if no face block for n."""
    if faces is None:
        return 0
    faces = np.asarray(faces)
    if n >= faces.shape[0]:
        return 0
    f = faces[n]
    return int(((f >= 0) & (f <= 1)).all(axis=1).sum())


def open_clip(stem):
    p = os.path.join(WREM, f"{stem}_dwpose.pkl")
    if os.path.exists(p):
        return pickle.load(open(p, 'rb'))
    raise FileNotFoundError(f"{stem} not in {WREM}")


def process_clip(stem, viz=False):
    d = open_clip(stem)
    os.makedirs(PKL_DIR, exist_ok=True)
    ref = front_table_scale(d)                  # STEP 1: front-table face size (near vs far tables)
    anchor = dealer_anchor(d, ref)              # STEP 2: dealer's central seat, DWPose only (no video/insightface)
    out = []
    cap = None; vw = None
    if viz:
        cap = cv2.VideoCapture(os.path.join(VC, f"{stem}.mp4"))
        import imageio
        os.makedirs(VIS_DIR, exist_ok=True)
        vw = imageio.get_writer(os.path.join(VIS_DIR, f"{stem}_mid.mp4"), fps=15,
                                codec='libx264', macro_block_size=None, ffmpeg_params=['-pix_fmt', 'yuv420p'])
    HAND_E = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],
              [0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20]]
    LIMBS = [[2,3],[2,6],[3,4],[4,5],[6,7],[7,8],[2,9],[9,10],[10,11],[2,12],[12,13],[13,14],[2,1],[1,15],[15,17],[1,16],[16,18]]
    picked = 0
    for fr in d:
        pose = fr['pose']; b = pose['bodies']
        H, W = fr['frame_dimensions']
        n = None; keep_idx = []
        if b is not None:
            cand = np.asarray(b['candidate']); sub = np.asarray(b['subset'])
            n = dealer_body(cand, sub, pose.get('faces'), anchor, W, H, ref)
            if n is not None and (valid_face_kpts(pose.get('faces'), n) < FACE_KPT_MIN
                                  or int((sub[n] >= 0).sum()) < MIN_BODY_KPTS):
                n = None                          # face not really visible OR sparse ghost body -> drop
            keep = link_hands(cand, sub, np.asarray(pose['hands']) if pose.get('hands') is not None else None, n, W, H)
            keep_idx = [j for j, _ in keep]
        if n is not None:
            picked += 1
            cand = np.asarray(b['candidate']); sub = np.asarray(b['subset'])
            hands = np.asarray(pose['hands']); isr = pose.get('hands_is_right', [])
            faces = np.asarray(pose['faces']) if pose.get('faces') is not None else None
            dc, ds = compact_body(cand, sub, n)
            out_fr = {'pose': {'bodies': {'candidate': dc, 'subset': ds},
                               'hands': hands[keep_idx] if keep_idx else hands[:0],
                               'hands_is_right': [isr[j] for j in keep_idx] if isr else [],
                               'faces': faces[n:n+1] if faces is not None and n < len(faces) else faces},
                      'frame_dimensions': fr['frame_dimensions']}
        else:
            out_fr = {'pose': {'bodies': None, 'hands': np.zeros((0, 21, 2)), 'hands_is_right': [], 'faces': None},
                      'frame_dimensions': fr['frame_dimensions'], 'dealer_not_found': True}
        out.append(out_fr)
        if vw is not None:
            ok, img = cap.read()
            if not ok: img = np.zeros((H, W, 3), np.uint8)
            if (img.shape[1], img.shape[0]) != (W, H): img = cv2.resize(img, (W, H))
            ov = img.copy()
            if b is not None:
                cand = np.asarray(b['candidate']); sub = np.asarray(b['subset'])
                for pn in range(sub.shape[0]):
                    col = (0, 255, 0) if pn == n else (0, 0, 255)     # dealer green, others red
                    for (u, v) in LIMBS:
                        ia, ib = int(sub[pn, u-1]), int(sub[pn, v-1])
                        if ia < 0 or ib < 0: continue
                        pa = cand[ia]*[W, H]; pb = cand[ib]*[W, H]
                        cv2.line(ov, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), col, 2)
                hh = np.asarray(pose['hands'])
                for j in range(hh.shape[0]):
                    col = (0, 255, 255) if j in keep_idx else (128, 128, 128)
                    pts = hh[j]*[W, H]
                    for e in HAND_E:
                        cv2.line(ov, tuple(pts[e[0]].astype(int)), tuple(pts[e[1]].astype(int)), col, 2)
            cv2.circle(ov, (int(anchor[0]), int(anchor[1])), 10, (255, 0, 255), 2)   # anchor (magenta)
            cv2.putText(ov, f"dealer={'FOUND' if n is not None else 'none'}  hands={len(keep_idx)}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0) if n is not None else (0, 0, 255), 2)
            vw.append_data(np.hstack([img, ov])[:, :, ::-1])
    if cap is not None: cap.release()
    if vw is not None: vw.close()
    op = os.path.join(PKL_DIR, f"{stem}_dealer.pkl"); pickle.dump(out, open(op, 'wb'))
    return len(out), picked, anchor, op


def multi_stems():
    df = pd.read_parquet(PARQ, columns=['base_name', 'single_person'])
    return [b[:-4] for b in df[df['single_person'] == False]['base_name']]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stem')
    ap.add_argument('--viz', action='store_true')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--viz_first', type=int, default=100)   # viz the first N clips (global index)
    a = ap.parse_args()
    if a.stem:
        n, picked, anchor, op = process_clip(a.stem, viz=a.viz)
        print(f"DONE {a.stem}: {n} frames, anchor=({anchor[0]:.0f},{anchor[1]:.0f}), "
              f"dealer found in {picked} ({100*picked/max(n,1):.1f}%) -> {op}", flush=True)
    else:
        stems = multi_stems()                                # global order (shared across shards)
        idx = {s: i for i, s in enumerate(stems)}
        mine = [s for i, s in enumerate(stems) if i % a.nshards == a.shard]
        ok = missing = 0
        for k, s in enumerate(mine):
            if os.path.exists(os.path.join(PKL_DIR, f"{s}_dealer.pkl")):
                continue
            if not os.path.exists(os.path.join(WREM, f"{s}_dwpose.pkl")):
                missing += 1; continue
            try:
                process_clip(s, viz=(idx[s] < a.viz_first))
                ok += 1
            except Exception as e:
                print(f"ERR {s}: {e}", flush=True)
            if (k + 1) % 50 == 0:
                print(f"shard{a.shard} {k+1}/{len(mine)} ok={ok} missing={missing}", flush=True)
        print(f"DONE shard{a.shard}: ok={ok} missing={missing}", flush=True)
