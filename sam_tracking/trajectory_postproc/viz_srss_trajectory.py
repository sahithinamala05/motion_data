"""Render trail-overlay viz for SRSS SAM trajectories (no SAM re-run).

Reads the trajectory JSONLs produced by sam_track_srss.py and draws, on a clip cut from
the raw session video, each tracked card's trail + current center + a card-sized box.
CPU-only (cv2 + ffmpeg); fast enough for a large sample.

Usage:
  python viz_srss_trajectory.py [N]     # N = number of trajectories to viz (default 100)
"""
import os, sys, glob, json, random, tempfile
from collections import defaultdict
import cv2

TRAJ_DIR = os.environ.get("TRAJ_DIR", "/home/ubuntu/us-west-3-fs/sahithi/hard_action_annotations/full_run/split_discard_gated_srss/sam_tracking/trajectories")
VIZ_DIR = os.environ.get("VIZ_DIR", "/home/ubuntu/us-west-3-fs/sahithi/hard_action_annotations/full_run/split_discard_gated_srss/sam_tracking/viz")
TRAJ_GLOB = os.environ.get("TRAJ_GLOB", "*_srss_trajectory.jsonl")
TRAJ_STRIP = os.environ.get("TRAJ_STRIP", "_srss_trajectory.jsonl")
RAW_VIDEO_DIR = "/home/ubuntu/us-west-3-fs/live_dealer_blackjack/raw/batch_01"
os.makedirs(VIZ_DIR, exist_ok=True)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
PAD = 4
W_OUT, H_OUT = 1280, 720
BOX_W, BOX_H = 76, 48          # nominal card-size box drawn around each tracked center
COLORS = [(0, 0, 255), (255, 0, 0), (0, 220, 0), (0, 200, 255)]  # BGR: red=card0, blue=card1


def session_path_and_offset(stem):
    parts = stem.split("_")
    base = f"{parts[0]}_{parts[1]}".replace("_", " ", 1)
    clip_start = int(parts[2])
    for name in (f"{base}.mp4", f"{parts[0]}_{parts[1]}.mp4"):
        p = os.path.join(RAW_VIDEO_DIR, name)
        if os.path.exists(p):
            return p, clip_start
    return None, None


def render(rec, stem, seg_k):
    sess, clip_start = session_path_and_offset(stem)
    if sess is None:
        return False
    sf, ef = rec["start_frame"], rec["end_frame"]     # local frames
    lo, hi = max(0, sf - PAD), ef + PAD
    cap = cv2.VideoCapture(sess)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, (clip_start - 1) + lo)
    out = os.path.join(VIZ_DIR, f"{stem}_seg{seg_k}.mp4")
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W_OUT, H_OUT))
    # index card trajectories by local frame
    byframe = defaultdict(list)   # local_frame -> [(oid, cx, cy)]
    for c in rec["cards"]:
        for t in c["trajectory"]:
            byframe[t[0]].append((c["card_index"], t[1], t[2]))
    trails = defaultdict(list)
    ss = rec.get("split_stats", {})
    verdict = "SPLIT" if rec.get("is_split") else "NOT SPLIT"
    vcol = (0, 220, 0) if rec.get("is_split") else (0, 0, 255)
    n = 0
    for i in range(hi - lo + 1):
        ret, img = cap.read()
        if not ret:
            break
        if img.shape[1] != W_OUT or img.shape[0] != H_OUT:
            img = cv2.resize(img, (W_OUT, H_OUT))
        f = lo + i
        for oid, cx, cy in byframe.get(f, []):
            trails[oid].append((int(cx), int(cy)))
        for oid in sorted(trails):
            color = COLORS[oid % len(COLORS)]
            pts = trails[oid]
            for k in range(1, len(pts)):
                cv2.line(img, pts[k - 1], pts[k], color, 2)
            if pts and any(o == oid for o, _, _ in byframe.get(f, [])):
                cx, cy = pts[-1]
                cv2.rectangle(img, (cx - BOX_W // 2, cy - BOX_H // 2),
                              (cx + BOX_W // 2, cy + BOX_H // 2), color, 2)
                cv2.circle(img, (cx, cy), 3, color, -1)
                cv2.putText(img, f"card{oid}", (cx - BOX_W // 2, cy - BOX_H // 2 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.putText(img, f"SAM3 srss  {verdict}  f{f} [{sf}-{ef}]", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, vcol, 2)
        cv2.putText(img, f"sep {ss.get('sep_start')}->{ss.get('sep_end')} max {ss.get('sep_max')} spread {ss.get('spread')}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, vcol, 2)
        vw.write(img)
        n += 1
    vw.release(); cap.release()
    if n == 0:
        os.remove(tmp)
        return False
    rc = os.system(f'ffmpeg -y -loglevel error -i "{tmp}" -c:v libx264 -pix_fmt yuv420p '
                   f'-movflags +faststart "{out}" 2>/dev/null')
    if rc == 0 and os.path.exists(out):
        os.remove(tmp)
    else:
        os.replace(tmp, out)
    return True


def main():
    files = sorted(glob.glob(os.path.join(TRAJ_DIR, TRAJ_GLOB)))
    random.seed(0)
    random.shuffle(files)
    files = files[:N]
    print(f"[info] rendering {len(files)} trail-overlay clips -> {VIZ_DIR}", flush=True)
    ok = 0
    for i, p in enumerate(files, 1):
        stem = os.path.basename(p).replace(TRAJ_STRIP, "")
        for k, line in enumerate(open(p)):
            rec = json.loads(line)
            if render(rec, stem, k):
                ok += 1
        if i % 20 == 0:
            print(f"  [{i}/{len(files)}] rendered={ok}", flush=True)
    print(f"\nDONE: {ok} clips in {VIZ_DIR}")


if __name__ == "__main__":
    main()
