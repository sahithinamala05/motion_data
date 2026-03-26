"""
Process instance_segments.json to extract call-for-action segments with
bounding boxes determined by index-finger-tip proximity to card centroids.

Pipeline:
  Stage 1 – duration   : drop call-for-action segments < MIN_SEGMENT_FRAMES
  Stage 2 – dwpose     : load hand tracking pkl; sample every POSE_STRIDE frames;
                         find index finger tip (keypoint 8) for both hands
  Stage 3 – card vote  : for each sampled frame with both hands detected, find
                         the nearest card centroid to each finger tip; choose the
                         card location that wins the most votes across all frames

Supports resuming after interruption: already-written annotation JSONs are
skipped on re-run.

Usage:
    python process_call_for_action_segments.py [--vis_count N] [--fail_vis_count N]
    --vis_count       0   no success vis (default); N=random N; -1=all
    --fail_vis_count  0   no failure vis (default); N=random N; -1=all
"""

import argparse
import json
import math
import os
import pickle
import random
import string
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ─── paths ────────────────────────────────────────────────────────────────────
SEGMENTS_JSON = (
    "/home/ubuntu/yifan/code/cleanpull/FACT_actseg/"
    "visualization_results_BH_0.7_2k5_unify_test/instance_segments.json"
)
CARDS_DIR = (
    "/home/ubuntu/us-west-3-fs/live_dealer_blackjack/"
    "roundcut/cards_results/good_quality_rounds/all_jsons"
)
PKL_DIR = Path(
    "/home/ubuntu/us-west-3-fs/live_dealer_blackjack/"
    "roundcut/dwpose/good_quality_rounds/dwpose"
)
RAW_VIDEO_DIR = "/home/ubuntu/yifan/code/FACT_actseg/Data_Filtering/filtered_videos"
OUTPUT_DIR = Path(__file__).parent / "output_cfa_segments_2k5"
FAILURE_VIS_DIR = OUTPUT_DIR / "failure_vis"
SUCCESS_VIS_DIR = OUTPUT_DIR / "success_vis"

# ─── constants ────────────────────────────────────────────────────────────────
IMG_W, IMG_H = 1280, 720
MIN_SEGMENT_FRAMES = 45
CFA_LABEL = "call for action"
POSE_STRIDE = 5            # sample every N frames within segment
INDEX_TIP_KP = 8           # hand keypoint index for index finger tip
UNDETECTED_THRESH = 0.01   # normalised coords below this → not detected


# ═══════════════════════════════════════════════════════════════════════════════
# Low-level helpers
# ═══════════════════════════════════════════════════════════════════════════════

def rand_id(n: int = 10) -> str:
    return "".join(random.choices(string.ascii_letters + string.digits, k=n))


def centre_dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class _NumpyCompatUnpickler(pickle.Unpickler):
    """Remap numpy._core ↔ numpy.core for cross-version pkl loading."""
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        elif module.startswith("numpy.core"):
            try:
                import numpy as np
                if not hasattr(np, "core"):
                    module = module.replace("numpy.core", "numpy._core")
            except ImportError:
                pass
        return super().find_class(module, name)


def load_pkl(path: str):
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except ModuleNotFoundError:
            pass
    with open(path, "rb") as f:
        return _NumpyCompatUnpickler(f).load()


def get_index_tip_px(
    frame_data: dict, hand_idx: int
) -> Optional[Tuple[int, int]]:
    """
    Returns (x, y) pixel coords of index finger tip (kp 8) for hand_idx
    (0=left, 1=right). Returns None if not detected.
    """
    H, W = frame_data["frame_dimensions"]
    hands = frame_data["pose"]["hands"]   # (2, 21, 2) normalised
    x_norm, y_norm = hands[hand_idx][INDEX_TIP_KP]
    if x_norm < UNDETECTED_THRESH and y_norm < UNDETECTED_THRESH:
        return None
    return int(x_norm * W), int(y_norm * H)


def load_card_detections(jsonl_path: str) -> Dict[int, List[dict]]:
    """Return {frame_id: [detections]} for frames with at least one detection."""
    out: Dict[int, List[dict]] = {}
    with open(jsonl_path) as f:
        for line in f:
            obj = json.loads(line)
            if obj["detections"]:
                out[obj["frame_id"]] = obj["detections"]
    return out


CARD_MIN_Y_PX = 450   # only consider cards whose centroid y > this value


def nearest_card(
    dets: List[dict], tip_px: Tuple[int, int]
) -> Optional[dict]:
    """Return the closest detection whose polygon_center y > CARD_MIN_Y_PX."""
    eligible = [d for d in dets if d["polygon_center"][0][1] > CARD_MIN_Y_PX]
    if not eligible:
        return None
    return min(eligible, key=lambda d: centre_dist(d["polygon_center"][0], tip_px))


def box_to_pct(box: List[int]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return (
        x1 / IMG_W * 100, y1 / IMG_H * 100,
        (x2 - x1) / IMG_W * 100, (y2 - y1) / IMG_H * 100,
    )


def make_bounding_box_entry(frame: int, det: dict) -> dict:
    x_pct, y_pct, w_pct, h_pct = box_to_pct(det["box"])
    return {
        "frame": frame,
        "labels": ["card"],
        "keyframes": [{
            "frame": frame,
            "x": x_pct, "y": y_pct, "width": w_pct, "height": h_pct,
            "polygon_center": det["polygon_center"][0],
            "box": det["box"],
            "conf": det.get("conf"),
            "rank": det.get("rank", ""),
            "suit": det.get("suit", ""),
        }],
        "id": rand_id(),
    }


def _base(seg: dict, video_name_base: str) -> dict:
    return {
        "video": video_name_base,
        "label": seg["label"],
        "start_frame": seg["start_frame"],
        "end_frame": seg["end_frame"],
        "duration_frames": seg["end_frame"] - seg["start_frame"] + 1,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Filtering stages
# ═══════════════════════════════════════════════════════════════════════════════

def stage1_duration(
    segments: List[dict], video_name_base: str
) -> Tuple[List[dict], List[dict]]:
    """Keep only call-for-action segments with >= MIN_SEGMENT_FRAMES frames."""
    passed, failed = [], []
    for seg in segments:
        if seg["label"] != CFA_LABEL:
            continue
        duration = seg["end_frame"] - seg["start_frame"] + 1
        if duration >= MIN_SEGMENT_FRAMES:
            passed.append(seg)
        else:
            failed.append({
                **_base(seg, video_name_base),
                "stage": "stage1_duration",
                "reason": f"duration={duration} < {MIN_SEGMENT_FRAMES}",
            })
    return passed, failed


def stage2_pose_sample(
    segments: List[dict],
    pose_data,
    card_detections: Optional[Dict[int, List[dict]]],
    video_name_base: str,
) -> Tuple[List[dict], List[dict]]:
    """
    For each segment, sample every POSE_STRIDE frames and collect
    (frame_idx, left_tip_px, right_tip_px, card_dets_at_frame).
    Enriches passing segments with "_samples" list.
    """
    passed, failed = [], []

    if pose_data is None:
        return [], [{
            **_base(seg, video_name_base),
            "stage": "stage2_no_pkl",
            "reason": "dwpose pkl not found",
        } for seg in segments]

    if card_detections is None:
        return [], [{
            **_base(seg, video_name_base),
            "stage": "stage2_no_card_jsonl",
            "reason": "card jsonl not found",
        } for seg in segments]

    for seg in segments:
        sf, ef = seg["start_frame"], seg["end_frame"]
        samples = []
        frame_indices = range(sf, ef + 1, POSE_STRIDE)

        for fi in frame_indices:
            if fi >= len(pose_data):
                continue
            fd = pose_data[fi]
            left_tip = get_index_tip_px(fd, 0)
            right_tip = get_index_tip_px(fd, 1)
            if left_tip is None or right_tip is None:
                continue
            dets = card_detections.get(fi, [])
            if not dets:
                # try adjacent frames within ±2
                for offset in [1, -1, 2, -2]:
                    dets = card_detections.get(fi + offset, [])
                    if dets:
                        break
            samples.append({
                "frame": fi,
                "left_tip": left_tip,
                "right_tip": right_tip,
                "card_dets": dets,
            })

        # need at least 1 sample with both hands + card dets
        valid_samples = [s for s in samples if s["card_dets"]]
        if not valid_samples:
            failed.append({
                **_base(seg, video_name_base),
                "stage": "stage2_no_valid_samples",
                "reason": (
                    f"{len(samples)} pose samples, none with card detections"
                ),
                "_samples": samples,
            })
            continue

        passed.append({**seg, "_samples": valid_samples})

    return passed, failed


CENTROID_SAME_CARD_THRESH = 20   # px – distance below which two centroids are the same card


def _card_key(det: dict) -> tuple:
    """Stable grid key for a detection (10 px grid)."""
    cx, cy = det["polygon_center"][0]
    return (round(cx / 10) * 10, round(cy / 10) * 10)


def stage3_card_vote(
    segments: List[dict], video_name_base: str
) -> Tuple[List[dict], List[dict]]:
    """
    Two-criterion selection across sampled frames:

    For every (frame, tip) pair find the nearest eligible card (y > CARD_MIN_Y_PX).

    C1  = card with the most votes (nearest-card wins)
    Dmin_one_hand = minimum distance at which C1 ever won a vote

    C2  = card with the single globally smallest distance to any tip in any frame
    Dmin = that distance

    C_final = C2  if  Dmin < Dmin_one_hand  AND  C1 != C2  (centroid > tolerance)
    C_final = C1  otherwise
    """
    passed, failed = [], []

    for seg in segments:
        samples = seg["_samples"]

        # key → [votes, det, last_frame, min_dist_when_voted]
        vote_map: Dict[tuple, list] = {}
        # global minimum over all (frame, tip, eligible_card) pairs
        global_min_dist = math.inf
        global_min_det = None
        global_min_frame = None

        for s in samples:
            dets = s["card_dets"]
            eligible = [d for d in dets if d["polygon_center"][0][1] > CARD_MIN_Y_PX]
            if not eligible:
                continue

            for tip in (s["left_tip"], s["right_tip"]):
                # ── vote: nearest eligible card ──────────────────────────
                best = min(eligible, key=lambda d: centre_dist(d["polygon_center"][0], tip))
                dist = centre_dist(best["polygon_center"][0], tip)
                key = _card_key(best)
                if key not in vote_map:
                    vote_map[key] = [0, best, s["frame"], math.inf]
                vote_map[key][0] += 1
                vote_map[key][2] = s["frame"]
                if dist < vote_map[key][3]:
                    vote_map[key][3] = dist

                # ── global minimum: ALL eligible cards for this tip ──────
                for det in eligible:
                    d = centre_dist(det["polygon_center"][0], tip)
                    if d < global_min_dist:
                        global_min_dist = d
                        global_min_det = det
                        global_min_frame = s["frame"]

        if not vote_map:
            failed.append({
                **_base(seg, video_name_base),
                "stage": "stage3_no_votes",
                "reason": "no card votes could be cast",
                "_samples": samples,
            })
            continue

        # ── C1: vote winner ───────────────────────────────────────────────
        best_key = max(vote_map, key=lambda k: vote_map[k][0])
        votes_c1, c1_det, c1_frame, dmin_one_hand = vote_map[best_key]

        # ── C2: globally closest card ─────────────────────────────────────
        c2_det = global_min_det
        c2_frame = global_min_frame

        # ── decide C_final ────────────────────────────────────────────────
        c1_c2_dist = centre_dist(
            c1_det["polygon_center"][0], c2_det["polygon_center"][0]
        )
        use_c2 = (
            global_min_dist < dmin_one_hand
            and c1_c2_dist > CENTROID_SAME_CARD_THRESH
        )

        chosen_det   = c2_det   if use_c2 else c1_det
        chosen_frame = c2_frame if use_c2 else c1_frame

        passed.append({
            **seg,
            "chosen_card": chosen_det,
            "chosen_frame": chosen_frame,
            "votes": votes_c1,
            "total_votes": sum(v[0] for v in vote_map.values()),
            "selection_method": "global_closest" if use_c2 else "vote_winner",
            "dmin_one_hand": dmin_one_hand,
            "global_min_dist": global_min_dist,
        })

    return passed, failed


# ═══════════════════════════════════════════════════════════════════════════════
# Visualisation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _load_segment_frames(video_name_base: str, sf: int, ef: int):
    try:
        import cv2
    except ImportError:
        return [], 25.0
    video_path = os.path.join(RAW_VIDEO_DIR, f"{video_name_base}.mp4")
    if not os.path.exists(video_path):
        return [], 25.0
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, sf)
    frames = []
    for _ in range(ef - sf + 1):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def _write_video(frames, out_path: str, fps: float = 25.0):
    import cv2, subprocess, tempfile
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if not frames:
        return
    h, w = frames[0].shape[:2]
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
    writer = cv2.VideoWriter(
        tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()
    subprocess.run(
        ["ffmpeg", "-y", "-i", tmp_path,
         "-vcodec", "libx264", "-crf", "23", "-preset", "fast",
         "-movflags", "+faststart", out_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
    )
    os.remove(tmp_path)


def visualize_success(vis_segments: List[dict], count: int, out_dir: str):
    """
    Render successful CFA segments with:
      - chosen card bbox (green)
      - index finger tips on sampled frames (left=cyan dot, right=magenta dot)
      - vote tally in corner
    """
    try:
        import cv2, numpy as np
    except ImportError:
        print("cv2 not available – skipping vis")
        return

    os.makedirs(out_dir, exist_ok=True)
    pool = list(vis_segments)
    if count != -1 and count < len(pool):
        pool = random.sample(pool, count)

    total = len(pool)
    for idx, info in enumerate(pool, 1):
        vname = info["video_name_base"]
        sf, ef = info["start_frame"], info["end_frame"]
        print(f"  success vis [{idx}/{total}] {vname} f{sf}-{ef}", flush=True)
        frames, fps = _load_segment_frames(vname, sf, ef)
        if not frames:
            frames = [np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)]

        chosen = info["chosen_card"]
        cx, cy = int(chosen["polygon_center"][0][0]), int(chosen["polygon_center"][0][1])
        x1, y1, x2, y2 = [int(v) for v in chosen["box"]]

        # build per-frame tip lookup
        tip_lookup: Dict[int, dict] = {
            s["frame"]: s for s in info.get("_samples", [])
        }

        for frame_offset, frame in enumerate(frames):
            fi = sf + frame_offset
            # draw chosen card
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)
            rank_suit = f"{chosen.get('rank','')}/{chosen.get('suit','')}"
            cv2.putText(frame, rank_suit, (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # draw finger tips on sampled frames
            if fi in tip_lookup:
                s = tip_lookup[fi]
                lx, ly = s["left_tip"]
                rx, ry = s["right_tip"]
                cv2.circle(frame, (lx, ly), 8, (255, 255, 0), -1)   # cyan = left
                cv2.circle(frame, (rx, ry), 8, (255, 0, 255), -1)   # magenta = right
                # draw lines from tips to chosen card centroid
                cv2.line(frame, (lx, ly), (cx, cy), (255, 255, 0), 1)
                cv2.line(frame, (rx, ry), (cx, cy), (255, 0, 255), 1)
                # draw all cards at this frame lightly
                for det in s["card_dets"]:
                    dx1, dy1, dx2, dy2 = [int(v) for v in det["box"]]
                    dcx, dcy = int(det["polygon_center"][0][0]), int(det["polygon_center"][0][1])
                    cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (80, 80, 255), 1)
                    cv2.circle(frame, (dcx, dcy), 3, (80, 80, 255), -1)

            method = info.get("selection_method", "vote_winner")
            dmin_oh = info.get("dmin_one_hand", 0)
            dmin_g  = info.get("global_min_dist", 0)
            cv2.putText(frame,
                        f"call-for-action  f{sf}-{ef}  [{method}]  "
                        f"votes:{info.get('votes',0)}/{info.get('total_votes',0)}  "
                        f"Dmin_C1:{dmin_oh:.1f}  Dmin_global:{dmin_g:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

        out_path = os.path.join(
            out_dir, f"cfa_{vname}_{sf}_{ef}_vis.mp4"
        )
        _write_video(frames, out_path, fps)

    print(f"Success vis ({total} clips) saved under: {out_dir}/")


def visualize_failures(all_failures: List[dict], count: int):
    if count == 0:
        return
    try:
        import cv2, numpy as np
    except ImportError:
        print("cv2 not available – skipping failure vis")
        return

    pool = list(all_failures)
    if count != -1 and count < len(pool):
        pool = random.sample(pool, count)

    total = len(pool)
    for idx, fc in enumerate(pool, 1):
        stage = fc["stage"]
        vname = fc["video"]
        sf, ef = fc["start_frame"], fc["end_frame"]
        print(f"  failure vis [{idx}/{total}] {vname} f{sf}-{ef} ({stage})", flush=True)
        out_path = str(
            FAILURE_VIS_DIR / stage / f"cfa_{vname}_{sf}_{ef}.mp4"
        )
        os.makedirs(str(FAILURE_VIS_DIR / stage), exist_ok=True)

        frames, fps = _load_segment_frames(vname, sf, ef)
        reason = fc.get("reason", "")

        if not frames:
            blank = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
            cv2.putText(blank, f"[{stage}] f{sf}-{ef}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 165, 0), 2)
            cv2.putText(blank, reason,
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            frames = [blank] * max(1, int(fps * 2))

        # overlay pose samples on relevant frames
        samples = fc.get("_samples", [])
        tip_lookup = {s["frame"]: s for s in samples}

        for frame_offset, frame in enumerate(frames):
            fi = sf + frame_offset
            cv2.putText(frame, f"[{stage}] f{sf}-{ef}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 165, 0), 2)
            cv2.putText(frame, reason,
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            if fi in tip_lookup:
                s = tip_lookup[fi]
                lx, ly = s["left_tip"]
                rx, ry = s["right_tip"]
                cv2.circle(frame, (lx, ly), 8, (255, 255, 0), -1)
                cv2.circle(frame, (rx, ry), 8, (255, 0, 255), -1)

        _write_video(frames, out_path, fps)

    print(f"Failure vis ({total} clips) saved under: {FAILURE_VIS_DIR}/")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def process(vis_count: int = 0, fail_vis_count: int = 0):
    with open(SEGMENTS_JSON) as f:
        all_segments: Dict[str, List[dict]] = json.load(f)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── detect already-processed videos for resume ─────────────────────────
    existing = {p.stem.replace("_annotations", "")
                for p in OUTPUT_DIR.glob("*_annotations.json")}
    if existing:
        print(f"Resuming: {len(existing)} videos already processed, skipping them.")

    all_failures: List[dict] = []
    vis_segments: List[dict] = []
    video_keys = list(all_segments.keys())
    total_videos = len(video_keys)
    n_skipped = 0
    n_processed = 0

    for vi, txt_key in enumerate(video_keys, 1):
        segments = all_segments[txt_key]
        video_name_base = txt_key.replace(".txt", "")
        video_name = video_name_base + ".mp4"

        # ── resume: skip already-written annotation JSONs ──────────────────
        if video_name_base in existing:
            n_skipped += 1
            continue

        # ── Stage 1: duration ──────────────────────────────────────────────
        s1_pass, s1_fail = stage1_duration(segments, video_name_base)
        all_failures.extend(s1_fail)

        if not s1_pass:
            # still write an empty annotation so resume skips this video
            _write_annotation(video_name_base, video_name, [])
            n_processed += 1
            if n_processed % 50 == 0:
                print(f"[{vi}/{total_videos}] {n_processed} processed, "
                      f"{n_skipped} skipped …", flush=True)
            continue

        print(f"[{vi}/{total_videos}] {video_name_base}  "
              f"({len(s1_pass)} CFA segs) …", flush=True)

        # load resources once per video
        pkl_path = PKL_DIR / f"{video_name_base}_dwpose.pkl"
        pose_data = load_pkl(str(pkl_path)) if pkl_path.exists() else None

        card_jsonl = os.path.join(CARDS_DIR, f"{video_name_base}_card.jsonl")
        card_dets = (
            load_card_detections(card_jsonl)
            if os.path.exists(card_jsonl) else None
        )

        # ── Stage 2: pose sampling ─────────────────────────────────────────
        s2_pass, s2_fail = stage2_pose_sample(
            s1_pass, pose_data, card_dets, video_name_base
        )
        all_failures.extend(s2_fail)

        # ── Stage 3: card vote ─────────────────────────────────────────────
        s3_pass, s3_fail = stage3_card_vote(s2_pass, video_name_base)
        all_failures.extend(s3_fail)

        # ── Build output timeline ──────────────────────────────────────────
        success_map = {
            (s["start_frame"], s["end_frame"]): s for s in s3_pass
        }

        timeline_segments = []
        for seg in segments:
            if seg["label"] != CFA_LABEL:
                continue
            sf, ef = seg["start_frame"], seg["end_frame"]
            duration = ef - sf + 1
            if duration < MIN_SEGMENT_FRAMES:
                continue
            bboxes = []
            if (sf, ef) in success_map:
                info = success_map[(sf, ef)]
                bboxes = [make_bounding_box_entry(info["chosen_frame"], info["chosen_card"])]
            timeline_segments.append({
                "start_frame": sf, "end_frame": ef,
                "labels": [CFA_LABEL],
                "meta_text": [],
                "duration_frames": duration,
                "id": rand_id(),
                "bounding_boxes": bboxes,
            })

        # write immediately so interrupted runs can resume
        _write_annotation(video_name_base, video_name, timeline_segments)

        for s in s3_pass:
            vis_segments.append({
                "video_name_base": video_name_base,
                "label": CFA_LABEL,
                "start_frame": s["start_frame"],
                "end_frame": s["end_frame"],
                "chosen_card": s["chosen_card"],
                "chosen_frame": s["chosen_frame"],
                "votes": s["votes"],
                "total_votes": s["total_votes"],
                "_samples": s["_samples"],
                "selection_method": s["selection_method"],
                "dmin_one_hand": s["dmin_one_hand"],
                "global_min_dist": s["global_min_dist"],
            })

        n_processed += 1

    # ── Stats ──────────────────────────────────────────────────────────────
    stage_counts: Dict[str, int] = {}
    for fc in all_failures:
        stage_counts[fc["stage"]] = stage_counts.get(fc["stage"], 0) + 1

    total_cfa = len(vis_segments) + len(all_failures)
    stats = {
        "total_cfa_segments": total_cfa,
        "successful": len(vis_segments),
        "failed_total": len(all_failures),
        "failed_by_stage": stage_counts,
        "skipped_resume": n_skipped,
    }
    with open(OUTPUT_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    def _clean(fc: dict) -> dict:
        return {k: v for k, v in fc.items() if not k.startswith("_")}

    with open(OUTPUT_DIR / "failure_cases.json", "w") as f:
        json.dump([_clean(fc) for fc in all_failures], f, indent=2)

    print(f"\nSummary: {len(vis_segments)} successful  /  "
          f"{len(all_failures)} failed  /  {total_cfa} total"
          f"  ({n_skipped} skipped from previous run)")
    for stage, cnt in sorted(stage_counts.items()):
        print(f"  {stage}: {cnt}")
    print(f"Stats   : {OUTPUT_DIR / 'stats.json'}")
    print(f"Failures: {OUTPUT_DIR / 'failure_cases.json'}")

    # ── Failure visualisation (optional) ───────────────────────────────────
    if fail_vis_count != 0:
        n = len(all_failures) if fail_vis_count == -1 else fail_vis_count
        print(f"\nGenerating {n} failure visualisations …")
        visualize_failures(all_failures, fail_vis_count)

    # ── Success visualisation (optional) ───────────────────────────────────
    if vis_count != 0:
        n = len(vis_segments) if vis_count == -1 else vis_count
        print(f"\nGenerating {n} success visualisations …")
        visualize_success(vis_segments, vis_count, str(SUCCESS_VIS_DIR))

    return vis_segments


def _write_annotation(video_name_base: str, video_name: str,
                      timeline_segments: List[dict]):
    """Write a single video's annotation JSON immediately (supports resume)."""
    record = {
        "video_name": video_name,
        "video_id": "",
        "video_path": "",
        "timeline_segments": timeline_segments,
    }
    out_path = OUTPUT_DIR / f"{video_name_base}_annotations.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)


# ─── entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vis_count", type=int, default=0,
        help="Success clips to render: 0=none (default), N=random N, -1=all",
    )
    parser.add_argument(
        "--fail_vis_count", type=int, default=0,
        help="Failure clips to render: 0=none (default), N=random N, -1=all",
    )
    args = parser.parse_args()
    process(vis_count=args.vis_count, fail_vis_count=args.fail_vis_count)
