"""Run the intra-pile post-processing on one round.

For a given round name:
1. Run the trained YOLO-OBB on a representative frame (default: the
   to_annotate prelabel frame for that round) -> per-seat OBB polygons.
2. Load the round's card detections jsonl.
3. Run intra_pile_util.label_round -> per-card metadata.
4. Save:
   - <out_dir>/<round>_per_card.json
   - <out_dir>/<round>_vis.jpg     (representative frame with seats + cards)
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, "/home/ubuntu/sahithi/motion-data-process/auto_label/episodes")
from intra_pile_util import label_round, point_in_obb, dedup_frame_detections
point_in_obb_local = point_in_obb  # alias used in draw_card_tracks

from ultralytics import YOLO

DEFAULT_MODEL = "/home/ubuntu/us-west-3-fs/sahithi/yolo_obb_runs/human_v2/weights/best.pt"
# Round-cut data: single file per round, card detections aligned to the round video,
# coords already in 1280x720 (the image space the OBB model predicts in).
DEFAULT_CARDS_DIR = "/home/ubuntu/us-west-3-fs/live_dealer_blackjack/roundcut/cards_results/good_quality_rounds/all_jsons"
DEFAULT_VIDEO_DIR = "/home/ubuntu/us-west-3-fs/live_dealer_blackjack/roundcut/good_quality_rounds"
DEFAULT_FRAMES_DIR = "/home/ubuntu/us-west-3-fs/sahithi/clus_anno/to_annotate"
DEFAULT_OUT_DIR = "/home/ubuntu/us-west-3-fs/sahithi/clus_anno/intra_pile_v1"

SEAT_NAMES = ["Dealer", "P1", "P2", "P3", "P4", "P5", "P6", "P7"]
SEAT_COLORS = [
    (128, 128, 128),  # Dealer - gray
    (60, 60, 220),    # P1 - red
    (60, 150, 240),   # P2 - orange
    (60, 220, 240),   # P3 - yellow
    (60, 200, 80),    # P4 - green
    (220, 200, 60),   # P5 - cyan
    (220, 100, 60),   # P6 - blue
    (180, 60, 200),   # P7 - magenta
]


def find_representative_frame(round_name: str, frames_dir: str) -> str | None:
    """Return path to an annotated frame for this round if it exists, else None."""
    matches = list(Path(frames_dir).glob(f"{round_name}_f*.png"))
    if not matches:
        return None
    # pick the one with the largest frame index (latest in the round)
    matches.sort(key=lambda p: int(p.stem.rsplit("_f", 1)[1]))
    return str(matches[-1])


def pick_rep_frame_by_detections(frames: list[dict], top_k: int = 20) -> list[tuple[int, int]]:
    """Return the top-k (round_fid, n_detections) tuples, sorted by n_detections desc.
    Caller can iterate and pick the first frame YOLO can actually read seats from
    (handles cases where the jsonl drifts out of sync with the chunk videos)."""
    rated = [(fr["frame_id"], len(fr.get("detections") or [])) for fr in frames]
    rated.sort(key=lambda x: (-x[1], x[0]))
    return rated[:top_k]


def extract_video_frame(round_name: str, round_fid: int, video_dir: str,
                        img_w: int, img_h: int):
    """Read the frame at `round_fid` (round-relative) from the chunk videos.
    Returns the BGR image resized to (img_w, img_h) or None on failure."""
    parts = round_name.rsplit("_", 2)
    g_start = int(parts[1])
    global_fid = g_start + round_fid
    chunks = find_round_chunks(round_name, video_dir)
    for chunk_start, chunk_end, path in chunks:
        if chunk_start <= global_fid <= chunk_end:
            cap = cv2.VideoCapture(path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, global_fid - chunk_start)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return None
            if (frame.shape[1], frame.shape[0]) != (img_w, img_h):
                frame = cv2.resize(frame, (img_w, img_h))
            return frame
    return None


def run_seat_obb(model, source, conf: float = 0.3):
    """Run YOLO-OBB on one image (path or np.ndarray). Returns dict {seat_id: [[x,y]*4]}."""
    results = model.predict(source=source, imgsz=1280, conf=conf, verbose=False)
    r = results[0]
    if r.obb is None or len(r.obb) == 0:
        return {}, r.orig_shape
    # r.obb.xyxyxyxy is (N, 4, 2) in pixel coords; r.obb.cls is (N,)
    polys = r.obb.xyxyxyxy.cpu().numpy()
    cls_ids = r.obb.cls.cpu().numpy().astype(int)
    confs = r.obb.conf.cpu().numpy()
    # If multiple detections for the same class, keep the highest-confidence one
    by_seat: dict[int, tuple[float, np.ndarray]] = {}
    for poly, cid, c in zip(polys, cls_ids, confs):
        prev = by_seat.get(int(cid))
        if prev is None or c > prev[0]:
            by_seat[int(cid)] = (float(c), poly)
    seat_obbs = {sid: by_seat[sid][1].tolist() for sid in by_seat}
    return seat_obbs, r.orig_shape


def draw_seat_obbs(img: np.ndarray, seat_obbs: dict[int, list]) -> None:
    for sid, corners in seat_obbs.items():
        col = SEAT_COLORS[sid]
        pts = np.array(corners, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], True, col, 2)
        cx, cy = np.mean(corners, axis=0).astype(int)
        cv2.putText(img, SEAT_NAMES[sid], (cx - 25, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)


def draw_raw_card_detections(
    img: np.ndarray, dets: list[dict],
    src_w: int, src_h: int, img_w: int, img_h: int,
) -> None:
    """Overlay the raw per-frame card detection boxes (gray thin) + rank/suit."""
    sx = img_w / src_w
    sy = img_h / src_h
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        x1, y1 = int(x1 * sx), int(y1 * sy)
        x2, y2 = int(x2 * sx), int(y2 * sy)
        cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 200), 1)
        rs = f"{d.get('rank','?')}{d.get('suit','?')}"
        cv2.putText(img, rs, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)


def draw_fan_vectors(
    img: np.ndarray, tracks: list, seat_obbs: dict, frame_detections: list,
) -> None:
    """For each seat with >=3 cards in this frame, draw arrows connecting
    consecutive card centroids (in ordinal order) and annotate sin(angle)
    between the fan vector v0 and the v0→v_i bend."""
    import math
    import math
    for sid, corners in seat_obbs.items():
        if sid == 0:
            continue
        seat_tracks = [t for t in tracks if t.get("seat") == sid]
        cards = []
        for det in frame_detections:
            pc = det.get("polygon_center")
            if pc is None:
                continue
            if isinstance(pc[0], list):
                pc = pc[0]
            r, s = det.get("rank", "?"), det.get("suit", "?")
            if (r, s) == ("CB", "CB"):
                continue
            if point_in_obb_local(pc, corners):
                cards.append((pc, r, s))
        if len(cards) < 3:
            continue
        # Match by (rank, suit) + closest centroid to disambiguate duplicates
        def card_ord(c):
            pc, r, s = c
            best_ord, best_d = 99, float('inf')
            for t in seat_tracks:
                if (t["rank"], t["suit"]) != (r, s):
                    continue
                tc = t["centers"][-1]
                d = math.hypot(pc[0] - tc[0], pc[1] - tc[1])
                if d < best_d:
                    best_d = d
                    best_ord = t.get("ordinal") or 99
            return best_ord
        cards.sort(key=card_ord)
        centers = [(int(c[0][0]), int(c[0][1])) for c in cards]
        col = SEAT_COLORS[sid]
        # v0 thick (the fan reference), v1 thinner so the two can be told apart
        cv2.arrowedLine(img, centers[0], centers[1], col, 3, tipLength=0.25)
        cv2.putText(img, "v0", (centers[1][0] - 16, centers[1][1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 2)
        cv2.arrowedLine(img, centers[1], centers[2], col, 2, tipLength=0.25)
        cv2.putText(img, "v1", (centers[2][0] - 16, centers[2][1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 2)
        if len(centers) >= 4:
            cv2.arrowedLine(img, centers[2], centers[3], col, 2, tipLength=0.25)
            cv2.putText(img, "v2", (centers[3][0] - 16, centers[3][1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 2)
        v0 = (centers[1][0] - centers[0][0], centers[1][1] - centers[0][1])
        v1 = (centers[2][0] - centers[1][0], centers[2][1] - centers[1][1])
        l0 = math.hypot(*v0); l1 = math.hypot(*v1)
        if l0 > 1 and l1 > 1:
            sin_a = (v0[0] * v1[1] - v0[1] * v1[0]) / (l0 * l1)
            cv2.putText(img, f"|sin|={abs(sin_a):.2f}",
                        (centers[2][0] + 8, centers[2][1] + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)


def draw_card_tracks(
    img: np.ndarray, tracks: list[dict], seat_obbs: dict[int, list],
    src_w: int, src_h: int, img_w: int, img_h: int,
    current_fid: int | None = None,
    frame_detections: list | None = None,
) -> None:
    """Visualization legend:
      - solid filled dot: card assigned via point-inside-OBB.
      - hollow ring + dotted line to seat centroid: assigned via nearest-OBB
        fallback.
      - red X: card track that couldn't be assigned to any seat.

    If `current_fid` is provided, draws each track at its centroid in that
    specific frame (or the closest observed frame). Otherwise falls back to
    the track's last observed centroid.
    """
    sx = img_w / src_w
    sy = img_h / src_h
    seat_centroids = {
        sid: tuple(np.mean(corners, axis=0).astype(int))
        for sid, corners in seat_obbs.items()
    }

    # If raw frame detections are provided: draw dots ONLY at those
    # centroids. Color by which seat OBB the centroid falls inside. No track
    # labels — purely a visualization of the frame's own data.
    if frame_detections is not None:
        for det in frame_detections:
            pc = det.get("polygon_center")
            if pc is None:
                continue
            if isinstance(pc[0], list):
                pc = pc[0]
            rank, suit = det.get("rank", "?"), det.get("suit", "?")
            if (rank, suit) == ("CB", "CB"):
                continue
            cx, cy = int(pc[0] * sx), int(pc[1] * sy)
            # Which seat OBB contains this centroid?
            seat = None
            for sid, corners in seat_obbs.items():
                if point_in_obb_local(pc, corners):
                    seat = sid
                    break
            if seat is None:
                cv2.drawMarker(img, (cx, cy), (0, 0, 255),
                               cv2.MARKER_TILTED_CROSS, markerSize=8, thickness=2)
            else:
                cv2.circle(img, (cx, cy), 4, SEAT_COLORS[seat], -1)
        return

    for t in tracks:
        if current_fid is not None:
            if not (t["first_frame"] <= current_fid <= t["last_frame"]):
                continue
            if "obs_frames" in t:
                obs_frames = t["obs_frames"]
                centers = t["centers"]
                idx = min(range(len(obs_frames)),
                          key=lambda i: abs(obs_frames[i] - current_fid))
                cx, cy = centers[idx]
            else:
                cx, cy = t["centers"][-1]
        else:
            cx, cy = t["centers"][-1]
        cx, cy = int(cx * sx), int(cy * sy)
        seat = t.get("seat")
        if seat is None:
            cv2.drawMarker(img, (cx, cy), (0, 0, 255), cv2.MARKER_TILTED_CROSS,
                           markerSize=10, thickness=2)
            cv2.putText(img, f"?{t['rank']}/{t['suit']}",
                        (cx + 6, cy - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 0, 255), 1)
            continue
        col = SEAT_COLORS[seat]
        via = t.get("seat_via") or ""
        is_fallback = via.startswith("nearest")
        if is_fallback:
            cv2.circle(img, (cx, cy), 6, col, 2)               # hollow ring
            # dotted line from point to seat centroid
            sc = seat_centroids.get(seat)
            if sc is not None:
                _dotted_line(img, (cx, cy), tuple(sc), col)
        else:
            cv2.circle(img, (cx, cy), 4, col, -1)              # solid dot
        seat_name = SEAT_NAMES[seat]
        ord_ = t.get("ordinal", "?")
        tag = f"{seat_name}:{ord_}"
        if t.get("double"):
            tag += " D"
        sp = t.get("split_pile", 0)
        if sp:
            tag += f" S{sp}"
        cv2.putText(img, tag, (cx + 6, cy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)


def _dotted_line(img, pt1, pt2, color, gap=4):
    x1, y1 = pt1
    x2, y2 = pt2
    dist = int(np.hypot(x2 - x1, y2 - y1))
    if dist == 0:
        return
    steps = max(1, dist // gap)
    for i in range(0, steps, 2):
        a = i / steps
        b = min(1.0, (i + 1) / steps)
        ax = int(x1 + a * (x2 - x1)); ay = int(y1 + a * (y2 - y1))
        bx = int(x1 + b * (x2 - x1)); by = int(y1 + b * (y2 - y1))
        cv2.line(img, (ax, ay), (bx, by), color, 1)


def round_video_path(round_name: str, video_dir: str) -> str:
    """Return the path to the pre-cut round mp4."""
    path = os.path.join(video_dir, f"{round_name}.mp4")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Round video not found: {path}")
    return path


def extract_video_frame_from_round(round_video: str, round_fid: int,
                                   img_w: int, img_h: int):
    """Read frame `round_fid` from the pre-cut round video. Returns BGR image
    resized to (img_w, img_h) or None on failure."""
    cap = cv2.VideoCapture(round_video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, round_fid)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    if (frame.shape[1], frame.shape[0]) != (img_w, img_h):
        frame = cv2.resize(frame, (img_w, img_h))
    return frame


def make_round_video(
    round_name: str,
    frames: list[dict],
    tracks: list[dict],
    seat_obbs: dict[int, list],
    video_dir: str,
    out_path: str,
    src_w: int = 1280, src_h: int = 720,
    img_w: int = 1280, img_h: int = 720,
) -> int:
    """Iterate every frame of the pre-cut round video, annotate with seat OBBs,
    raw card detections, and per-frame active track labels. Write to out_path."""
    in_path = round_video_path(round_name, video_dir)
    cap = cv2.VideoCapture(in_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Per-frame deduped raw detections lookup
    frame_dets = {fr["frame_id"]: dedup_frame_detections(fr.get("detections") or [])
                  for fr in frames}
    # Per-frame active tracks lookup: round_fid -> list of (track, center, box)
    frame_tracks: dict[int, list] = {}
    for t in tracks:
        for fid, c, b in zip(t["obs_frames"], t["centers"], t["boxes"]):
            frame_tracks.setdefault(fid, []).append((t, c, b))

    # OpenCV's H.264 writer requires v4l2m2m hardware which isn't available
    # here. Write mp4v first, then transcode to H.264 with ffmpeg below.
    tmp_out = out_path + ".tmp.mp4"
    writer = cv2.VideoWriter(tmp_out, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (img_w, img_h))
    sx = img_w / src_w
    sy = img_h / src_h

    round_fid = -1
    written = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        round_fid += 1
        if (frame.shape[1], frame.shape[0]) != (img_w, img_h):
            frame = cv2.resize(frame, (img_w, img_h))

        # Layer 1: raw card detections at this frame
        draw_raw_card_detections(frame, frame_dets.get(round_fid, []),
                                 src_w, src_h, img_w, img_h)
        # Layer 2: seat OBBs (fixed across the round)
        draw_seat_obbs(frame, seat_obbs)
        # Layer 3: active tracks at this frame
        for t, c, b in frame_tracks.get(round_fid, []):
            cx, cy = int(c[0] * sx), int(c[1] * sy)
            seat = t.get("seat")
            if seat is None:
                cv2.drawMarker(frame, (cx, cy), (0, 0, 255),
                               cv2.MARKER_TILTED_CROSS, markerSize=10, thickness=2)
                continue
            col = SEAT_COLORS[seat]
            via = t.get("seat_via") or ""
            if via.startswith("nearest"):
                cv2.circle(frame, (cx, cy), 6, col, 2)
            else:
                cv2.circle(frame, (cx, cy), 4, col, -1)
            tag = f"{SEAT_NAMES[seat]}:{t.get('ordinal','?')}"
            if t.get("double"): tag += " D"
            if t.get("split_pile"): tag += f" S{t['split_pile']}"
            cv2.putText(frame, tag, (cx + 6, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
        # Frame counter
        cv2.putText(frame, f"f={round_fid}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(frame)
        written += 1

    cap.release()
    writer.release()
    # Transcode mp4v → H.264 (libx264) so the file plays in standard viewers.
    import subprocess
    cmd = ["/usr/bin/ffmpeg", "-y", "-loglevel", "error",
           "-i", tmp_out, "-c:v", "libx264", "-preset", "veryfast",
           "-pix_fmt", "yuv420p", out_path]
    rc = subprocess.run(cmd).returncode
    if rc == 0:
        os.remove(tmp_out)
    else:
        # Transcode failed; keep mp4v output under final name
        os.replace(tmp_out, out_path)
    return written


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--round", required=True, help="Round name (no extension)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--cards_dir", default=DEFAULT_CARDS_DIR)
    p.add_argument("--frames_dir", default=DEFAULT_FRAMES_DIR)
    p.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--video_dir", default=DEFAULT_VIDEO_DIR,
                   help="Source video chunks directory (used when --video).")
    p.add_argument("--video", action="store_true",
                   help="Also render an annotated mp4 across the whole round.")
    # Round-cut card detections live in 1280x720 already; the chunked roundwise
    # detections were in 1920x1080. Override --src_w/--src_h for the latter.
    p.add_argument("--src_w", type=int, default=1280)
    p.add_argument("--src_h", type=int, default=720)
    p.add_argument("--img_w", type=int, default=1280)
    p.add_argument("--img_h", type=int, default=720)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cards_path = Path(args.cards_dir) / f"{args.round}_card.jsonl"
    if not cards_path.exists():
        raise SystemExit(f"Card jsonl not found: {cards_path}")
    frames = [json.loads(l) for l in open(cards_path)]
    print(f"Loaded {len(frames)} frames of card detections for round {args.round}")

    round_video = round_video_path(args.round, args.video_dir)
    print(f"Round video: {round_video}")

    # Pick the frame with the most card detections from the jsonl, extract it
    # from video, and run YOLO. If YOLO doesn't find any seats (e.g., the jsonl
    # has drifted out of sync with the video around the chosen frame), try the
    # next-best candidate. As a last resort, fall back to the annotated frame.
    model = YOLO(args.model)
    TOP_K = 10        # how many max-detection frames to probe
    EARLY_EXIT = 8    # accept early if we hit this many seats
    candidates = pick_rep_frame_by_detections(frames, top_k=TOP_K)
    best = None       # (n_seats, fid, img, seat_obbs)
    for fid, n in candidates:
        img = extract_video_frame_from_round(round_video, fid,
                                              args.img_w, args.img_h)
        if img is None:
            continue
        sobbs, _ = run_seat_obb(model, img)
        print(f"  probe round_fid={fid} (jsonl_dets={n}, yolo_seats={len(sobbs)})")
        if best is None or len(sobbs) > best[0]:
            best = (len(sobbs), fid, img, sobbs)
            if len(sobbs) >= EARLY_EXIT:
                break
    rep_fid, rep_img, seat_obbs = (None, None, {}) if best is None else (best[1], best[2], best[3])
    if rep_img is not None and len(seat_obbs) > 0:
        print(f"Representative frame: round_fid={rep_fid} (yolo_seats={len(seat_obbs)})")
    if rep_img is None:
        # Last-resort fallback: use the to_annotate frame
        frame_path = find_representative_frame(args.round, args.frames_dir)
        if frame_path is None:
            raise SystemExit(
                f"Couldn't find any usable rep frame for {args.round}"
            )
        rep_fid = int(Path(frame_path).stem.rsplit("_f", 1)[1])
        rep_img = cv2.imread(frame_path)
        seat_obbs, _ = run_seat_obb(model, rep_img)
        print(f"Fell back to annotated frame {frame_path} (round_fid={rep_fid})")
    print(f"Detected seats: {sorted(seat_obbs)} ({len(seat_obbs)}/8)")

    tracks = label_round(
        frames, seat_obbs,
        src_w=args.src_w, src_h=args.src_h,
        img_w=args.img_w, img_h=args.img_h,
        rep_fid=rep_fid,
    )
    print(f"Built {len(tracks)} card tracks")

    # Summarize
    by_seat: dict[int, int] = {}
    doubles = 0
    splits = 0
    unassigned = 0
    for t in tracks:
        s = t.get("seat")
        if s is None:
            unassigned += 1
            continue
        by_seat[s] = by_seat.get(s, 0) + 1
        if t.get("double"): doubles += 1
        if t.get("split_pile"): splits += 1
    print(f"Cards per seat: {dict(sorted(by_seat.items()))}")
    print(f"Doubles flagged: {doubles}   Splits flagged: {splits}   Unassigned: {unassigned}")

    # Save JSON (trim heavy fields for readability)
    out_json = out_dir / f"{args.round}_per_card.json"
    serialized = []
    for t in tracks:
        serialized.append({
            "track_id": t["track_id"],
            "rank": t["rank"], "suit": t["suit"],
            "seat": t.get("seat"),
            "seat_name": SEAT_NAMES[t["seat"]] if t.get("seat") is not None else None,
            "seat_via": t.get("seat_via"),
            "ordinal": t.get("ordinal"),
            "double": t.get("double", False),
            "split_pile": t.get("split_pile", 0),
            "first_frame": t["first_frame"],
            "last_frame": t["last_frame"],
            "first_center": t["first_center"],
            "last_center": t["centers"][-1],
            "n_observations": len(t["centers"]),
            "mean_conf": float(np.mean(t["confs"])),
        })
    with open(out_json, "w") as f:
        json.dump({
            "round": args.round,
            "representative_frame": f"round_fid={rep_fid}",
            "seat_obbs": seat_obbs,
            "tracks": serialized,
        }, f, indent=2)
    print(f"Wrote {out_json}")

    # Save visualization on the rep frame extracted from video
    img = rep_img.copy()
    rep_dets = next((fr.get("detections") or [] for fr in frames
                     if fr.get("frame_id") == rep_fid), [])
    rep_dets = dedup_frame_detections(rep_dets)
    draw_raw_card_detections(img, rep_dets,
                             args.src_w, args.src_h, args.img_w, args.img_h)
    draw_seat_obbs(img, seat_obbs)
    draw_card_tracks(img, tracks, seat_obbs, args.src_w, args.src_h,
                     args.img_w, args.img_h, current_fid=rep_fid,
                     frame_detections=rep_dets)
    draw_fan_vectors(img, tracks, seat_obbs, rep_dets)
    vis_path = out_dir / f"{args.round}_vis.jpg"
    cv2.imwrite(str(vis_path), img)
    print(f"Wrote {vis_path}")

    if args.video:
        video_path = out_dir / f"{args.round}_vis.mp4"
        n = make_round_video(
            args.round, frames, tracks, seat_obbs,
            args.video_dir, str(video_path),
            args.src_w, args.src_h, args.img_w, args.img_h,
        )
        print(f"Wrote {video_path} ({n} frames)")


if __name__ == "__main__":
    main()
