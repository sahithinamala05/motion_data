"""
Process ActionFormer tap predictions with DWPose + card detection verification.

Pipeline:
  Stage 1 – Fingertip motion  : check DWPose fingertips for up-down tap motion
  Stage 2 – Nearest card      : find card closest to the tap point on table
  Stage 3 – Negative card     : reject if new cards appear (likely a hit, not tap)

Usage:
    python process_tap_segments.py [--vis_count N] [--fail_vis_count N]
                                   [--max_videos N] [--workers N]
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import pickle
import random
import string
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ─── Paths ───────────────────────────────────────────────────────────────────
PREDICTIONS_JSON = (
    "/home/ubuntu/us-west-3-fs/sahithi/hard_action_annotations/"
    "ambiguous_actions_all/predictions.json"
)
DWPOSE_DIR = (
    "/home/ubuntu/us-west-3-fs/sahithi/hard_action_annotations/dwpose_roundwise"
)
CARDS_DIR = (
    "/home/ubuntu/us-west-3-fs/sahithi/hard_action_annotations/card_detections_roundwise"
)
RAW_VIDEO_DIR = "/home/ubuntu/us-west-3-fs/live_dealer_blackjack/raw/batch_01"
OUTPUT_DIR = Path(
    "/home/ubuntu/us-west-3-fs/sahithi/hard_action_annotations/full_run/tap_final_2"
)
SUCCESS_VIS_DIR = OUTPUT_DIR / "success_vis"
FAILURE_VIS_DIR = OUTPUT_DIR / "failure_vis"

# ─── Constants ───────────────────────────────────────────────────────────────
IMG_W, IMG_H = 1280, 720
SCALE = 1280 / 1920  # card detections are 1080p, scale to 720p

# DWPose hand fingertip indices (per hand, 21 keypoints each)
FINGERTIP_INDICES = [8, 12, 16, 20]  # index, middle, ring, pinky (no thumb)

# Tap motion thresholds
MIN_Y_DIP = 0.02        # min normalized y-dip for a fingertip to count as tap motion
MIN_SCORE_THRESH = 0.3  # ActionFormer score threshold
CARD_MATCH_DIST = 150   # max px distance from tap point to nearest card
TABLE_MIN_Y_PX = 450    # tap point must be on the table (y >= this in pixels)
CARD_MIN_Y_PX = 480     # only match cards in player zone (y >= this)


# ─── Utilities ───────────────────────────────────────────────────────────────

def rand_id(n=10):
    return "".join(random.choices(string.ascii_letters + string.digits, k=n))


def _scale_det(det):
    cx, cy = det["polygon_center"][0]
    x1, y1, x2, y2 = det["box"]
    return {
        **det,
        "polygon_center": [[cx * SCALE, cy * SCALE]],
        "box": [int(x1 * SCALE), int(y1 * SCALE),
                int(x2 * SCALE), int(y2 * SCALE)],
    }


def load_card_detections(jsonl_path):
    out = {}
    with open(jsonl_path) as f:
        for line in f:
            obj = json.loads(line)
            if obj["detections"]:
                out[obj["frame_id"]] = [_scale_det(d) for d in obj["detections"]]
    return out


def load_dwpose(vname):
    pkl_path = os.path.join(DWPOSE_DIR, f"{vname}_dwpose.pkl")
    if not os.path.exists(pkl_path):
        return None
    with open(pkl_path, "rb") as f:
        return pickle.load(f)



# ─── Stage 1: Fingertip motion detection ─────────────────────────────────────

def find_tap_motion(dwpose_data, sf, ef):
    """
    Check all fingertips on both hands for up-down tap motion.
    Returns (tap_frame, tap_x_px, tap_y_px, hand_idx, fingertip_idx, y_dip)
    or None if no tap motion found.

    Coordinates in dwpose are normalized 0-1.
    """
    n_frames = ef - sf + 1
    if n_frames < 3:
        return None

    best_dip = 0
    best_result = None

    # Check both hands (0=right, 1=left in dwpose)
    for hand_idx in range(2):
        for ft_idx in FINGERTIP_INDICES:
            # Extract y-trajectory for this fingertip across the segment
            # Only consider frames where fingertip is on the table area
            ys = []
            xs = []
            frame_indices = []
            valid = True
            table_y_norm = TABLE_MIN_Y_PX / IMG_H  # normalized threshold
            for rel_f in range(n_frames):
                abs_f = sf + rel_f
                if abs_f >= len(dwpose_data):
                    valid = False
                    break
                frame_data = dwpose_data[abs_f]
                pose = frame_data.get("pose", {})
                hands = pose.get("hands", None)
                if hands is None or hands.shape[0] <= hand_idx:
                    valid = False
                    break
                hand_kps = hands[hand_idx]  # shape (21, 2)
                if ft_idx >= hand_kps.shape[0]:
                    valid = False
                    break
                fy = float(hand_kps[ft_idx][1])
                # Only track fingertip when it's in the table area
                if fy < table_y_norm:
                    continue
                xs.append(float(hand_kps[ft_idx][0]))
                ys.append(fy)
                frame_indices.append(rel_f)

            if not valid or len(ys) < 3:
                continue

            # Look for downward motion: y increases = fingertip moves toward table
            # Accept both full dip (down-up) and downward-only motion
            min_y = min(ys)
            max_y = max(ys)
            y_range = max_y - min_y

            if y_range < MIN_Y_DIP:
                continue

            # Find the first local peak with sufficient down-AND-up dip
            # Skip tiny noise bumps — only accept peaks with real tap motion
            found_peak = False
            peak_idx = -1
            for pi in range(1, len(ys) - 1):
                if ys[pi] >= ys[pi - 1] and ys[pi] >= ys[pi + 1]:
                    y_before = min(ys[:pi])
                    y_after = min(ys[pi + 1:])
                    dip_down = ys[pi] - y_before
                    dip_up = ys[pi] - y_after
                    if dip_down >= MIN_Y_DIP and dip_up >= MIN_Y_DIP:
                        peak_idx = pi
                        found_peak = True
                        break

            if not found_peak:
                continue

            peak_y = ys[peak_idx]
            y_before = min(ys[:peak_idx])
            y_after = min(ys[peak_idx + 1:])
            dip_down = peak_y - y_before
            dip_up = peak_y - y_after
            total_dip = dip_down + dip_up
            if total_dip > best_dip:
                best_dip = total_dip
                tap_frame = sf + frame_indices[peak_idx]
                # Convert normalized coords to pixel coords
                tap_x_px = xs[peak_idx] * IMG_W
                tap_y_px = ys[peak_idx] * IMG_H
                best_result = (tap_frame, tap_x_px, tap_y_px,
                               hand_idx, ft_idx, total_dip)

    return best_result


# ─── Stage 2: Nearest card matching ──────────────────────────────────────────

def find_nearest_card(card_dets, frame_id, tap_x, tap_y):
    """Find the card detection closest to the tap point."""
    if frame_id not in card_dets:
        # Search nearby frames
        for offset in range(1, 10):
            for f in [frame_id + offset, frame_id - offset]:
                if f in card_dets:
                    frame_id = f
                    break
            else:
                continue
            break

    if frame_id not in card_dets:
        return None, None, None

    dets = card_dets[frame_id]
    # Only consider cards in player zone (y >= 450px)
    dets = [d for d in dets if d["polygon_center"][0][1] >= CARD_MIN_Y_PX]
    if not dets:
        return None, None, None

    best_det = None
    best_dist = float("inf")
    for det in dets:
        cx, cy = det["polygon_center"][0]
        dist = math.hypot(cx - tap_x, cy - tap_y)  # euclidean distance
        if dist < best_dist:
            best_dist = dist
            best_det = det

    if best_dist > CARD_MATCH_DIST:
        return None, best_dist, None

    return best_det, best_dist, frame_id


# ─── Stage 3: New card detection (reject hits) ──────────────────────────────

def _count_player_cards(card_dets, frame_id):
    """Count cards in the player zone (y >= CARD_MIN_Y_PX) at a given frame."""
    if frame_id not in card_dets:
        # Search nearby frames
        for offset in range(1, 10):
            for f in [frame_id + offset, frame_id - offset]:
                if f in card_dets:
                    frame_id = f
                    break
            else:
                continue
            break
    if frame_id not in card_dets:
        return -1  # unknown
    dets = card_dets[frame_id]
    return sum(1 for d in dets if d["polygon_center"][0][1] >= CARD_MIN_Y_PX)


def check_new_card(card_dets, sf, ef):
    """Return True if a new card appears in the player zone during the segment."""
    start_count = _count_player_cards(card_dets, sf)
    end_count = _count_player_cards(card_dets, ef)
    if start_count < 0 or end_count < 0:
        return False  # can't tell, don't reject
    return end_count > start_count


# ─── Video helpers ────────────────────────────────────────────────────────────

def _get_video_path(vname):
    parts = vname.split("_")
    base = f"{parts[0]}_{parts[1]}".replace("_", " ", 1)
    for name in (f"{base}.mp4", f"{parts[0]}_{parts[1]}.mp4"):
        path = os.path.join(RAW_VIDEO_DIR, name)
        if os.path.exists(path):
            return path, int(parts[2])
    return None, None


def _load_segment_frames(vname, sf, ef):
    path, clip_start = _get_video_path(vname)
    if path is None:
        return [], 25.0
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, (clip_start - 1) + sf)
    frames = []
    for _ in range(ef - sf + 1):
        ret, frame = cap.read()
        if not ret:
            break
        if frame.shape[1] != IMG_W or frame.shape[0] != IMG_H:
            frame = cv2.resize(frame, (IMG_W, IMG_H))
        frames.append(frame)
    cap.release()
    return frames, fps


def _write_video(frames, out_path, fps=25.0):
    if not frames:
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    h, w = frames[0].shape[:2]
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
    writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    subprocess.run(
        ["/usr/bin/ffmpeg", "-y", "-i", tmp_path, "-vcodec", "libx264",
         "-crf", "23", "-preset", "fast", "-movflags", "+faststart", out_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
    )
    os.remove(tmp_path)


# ─── Visualization ────────────────────────────────────────────────────────────

def visualize_failures(failures, count, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    pool = list(failures)
    if count != -1 and count < len(pool):
        pool = random.sample(pool, count)

    for idx, info in enumerate(pool, 1):
        vname = info["vname"]
        sf, ef = info["start_frame"], info["end_frame"]
        reason = info.get("reason", "unknown")
        score = info.get("score", 0)

        print(f"  fail vis [{idx}/{len(pool)}] {vname} f{sf}-{ef} ({reason})", flush=True)

        frames, fps = _load_segment_frames(vname, sf, ef)
        if not frames:
            continue

        for i, frame in enumerate(frames):
            cv2.line(frame, (0, TABLE_MIN_Y_PX), (IMG_W, TABLE_MIN_Y_PX),
                     (0, 165, 255), 2)
            cv2.putText(frame, f"FAIL [{reason}] score={score:.2f} f{sf+i}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        out_path = os.path.join(out_dir, f"fail_{vname}_{sf}_{ef}.mp4")
        _write_video(frames, out_path, fps)

    print(f"Failure vis ({len(pool)} clips) saved to: {out_dir}/")


def visualize_success(segments, count, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    pool = list(segments)
    if count != -1 and count < len(pool):
        pool = random.sample(pool, count)

    for idx, info in enumerate(pool, 1):
        vname = info["vname"]
        sf, ef = info["start_frame"], info["end_frame"]
        tap_frame = info["tap_frame"]
        tap_x, tap_y = info["tap_x"], info["tap_y"]
        card = info.get("nearest_card")
        score = info.get("score", 0)

        print(f"  vis [{idx}/{len(pool)}] {vname} f{sf}-{ef} tap@f{tap_frame}", flush=True)

        frames, fps = _load_segment_frames(vname, sf, ef)
        if not frames:
            continue

        for i, frame in enumerate(frames):
            fi = sf + i
            # Draw table line
            cv2.line(frame, (0, TABLE_MIN_Y_PX), (IMG_W, TABLE_MIN_Y_PX),
                     (0, 165, 255), 2)
            cv2.putText(frame, f"table y>={TABLE_MIN_Y_PX}", (5, TABLE_MIN_Y_PX - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

            # Draw tap point
            tx, ty = int(tap_x), int(tap_y)
            color = (0, 255, 0) if fi == tap_frame else (100, 200, 100)
            cv2.circle(frame, (tx, ty), 8, color, -1)
            cv2.putText(frame, "TAP", (tx + 10, ty - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw nearest card bbox
            if card:
                x1, y1, x2, y2 = [int(v) for v in card["box"]]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                rank = card.get("rank", "?")
                suit = card.get("suit", "?")
                cv2.putText(frame, f"{rank}{suit}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            cv2.putText(frame, f"TAP f{fi} (score={score:.2f}) seg {sf}-{ef}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out_path = os.path.join(out_dir, f"tap_{vname}_{sf}_{ef}.mp4")
        _write_video(frames, out_path, fps)

    print(f"Success vis ({len(pool)} clips) saved to: {out_dir}/")


# ─── Per-video worker (runs in subprocess) ───────────────────────────────────

def _process_one_video(args):
    vname, segs = args
    passed = 0
    dropped_no_dwpose = 0
    dropped_no_motion = 0
    dropped_no_card = 0
    dropped_new_card = 0
    pass_pool = []
    fail_pool = []

    dwpose = load_dwpose(vname)
    if dwpose is None:
        dropped_no_dwpose = len(segs)
        fail_pool = [{**seg, "reason": "no dwpose"} for seg in segs]
        return {"passed": passed, "dropped_no_dwpose": dropped_no_dwpose,
                "dropped_no_motion": 0, "dropped_no_card": 0,
                "dropped_new_card": 0, "pass_pool": [], "fail_pool": fail_pool}

    card_jsonl = os.path.join(CARDS_DIR, f"{vname}_card.jsonl")
    card_dets = load_card_detections(card_jsonl) if os.path.exists(card_jsonl) else {}

    for seg in segs:
        sf, ef = seg["start_frame"], seg["end_frame"]

        # Stage 1: Check fingertip motion
        tap_result = find_tap_motion(dwpose, sf, ef)
        if tap_result is None:
            dropped_no_motion += 1
            fail_pool.append({**seg, "reason": "no tap motion detected"})
            continue

        tap_frame, tap_x, tap_y, hand_idx, ft_idx, y_dip = tap_result

        if tap_y < TABLE_MIN_Y_PX:
            dropped_no_motion += 1
            fail_pool.append({**seg, "reason": f"tap point y={tap_y:.0f} above table ({TABLE_MIN_Y_PX})"})
            continue

        # Stage 2: Find nearest card
        nearest_card, card_dist, card_frame = find_nearest_card(
            card_dets, tap_frame, tap_x, tap_y)

        if nearest_card is None:
            dropped_no_card += 1
            fail_pool.append({**seg, "reason": f"no card within {CARD_MATCH_DIST}px"})
            continue

        # Stage 3: Reject if new card appears (likely a hit, not tap)
        if check_new_card(card_dets, sf, ef):
            dropped_new_card += 1
            fail_pool.append({**seg, "reason": "new card appeared (hit, not tap)"})
            continue

        # Passed all stages
        passed += 1
        pass_pool.append({
            **seg,
            "tap_frame": tap_frame,
            "tap_x": tap_x,
            "tap_y": tap_y,
            "hand_idx": hand_idx,
            "fingertip_idx": ft_idx,
            "y_dip": y_dip,
            "nearest_card": nearest_card,
            "card_dist": card_dist,
            "card_frame": card_frame,
        })

    return {"passed": passed, "dropped_no_dwpose": dropped_no_dwpose,
            "dropped_no_motion": dropped_no_motion, "dropped_no_card": dropped_no_card,
            "dropped_new_card": dropped_new_card,
            "pass_pool": pass_pool, "fail_pool": fail_pool}


# ─── Main pipeline ────────────────────────────────────────────────────────────

def process(vis_count=0, fail_vis_count=0, max_videos=0, **kwargs):
    print(f"Loading predictions from: {PREDICTIONS_JSON}", flush=True)
    with open(PREDICTIONS_JSON) as f:
        preds = json.load(f)

    # Collect all tap detections
    tap_segments = []
    for vname, v in preds.items():
        for det in v["detections"]:
            if det["label"] == "tap" and det["score"] >= MIN_SCORE_THRESH:
                tap_segments.append({
                    "vname": vname,
                    "start_frame": det["start_frame"],
                    "end_frame": det["end_frame"],
                    "score": det["score"],
                })

    print(f"Total tap predictions: {len(tap_segments)}", flush=True)

    if max_videos:
        # Limit to first N unique videos
        seen = set()
        limited = []
        for seg in tap_segments:
            if seg["vname"] not in seen:
                seen.add(seg["vname"])
            if len(seen) <= max_videos:
                limited.append(seg)
        tap_segments = limited
        print(f"Limited to {len(tap_segments)} taps from {max_videos} videos", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total = len(tap_segments)
    passed = 0
    dropped_no_dwpose = 0
    dropped_no_motion = 0
    dropped_no_card = 0
    dropped_new_card = 0
    pass_pool = []
    fail_pool = []

    # Group by video for efficient loading
    by_video = defaultdict(list)
    for seg in tap_segments:
        by_video[seg["vname"]].append(seg)

    video_items = list(by_video.items())
    num_workers = kwargs.get("workers", 8)
    n_videos = len(video_items)

    with mp.Pool(num_workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_process_one_video, video_items), 1):
            passed += res["passed"]
            dropped_no_dwpose += res["dropped_no_dwpose"]
            dropped_no_motion += res["dropped_no_motion"]
            dropped_no_card += res["dropped_no_card"]
            dropped_new_card += res["dropped_new_card"]
            pass_pool.extend(res["pass_pool"])
            fail_pool.extend(res["fail_pool"])
            if i % 500 == 0 or i == n_videos:
                print(f"[{i}/{n_videos}] passed={passed} no_motion={dropped_no_motion} "
                      f"no_card={dropped_no_card} new_card={dropped_new_card}", flush=True)

    # Stats
    stats = {
        "total": total,
        "passed": passed,
        "dropped_no_dwpose": dropped_no_dwpose,
        "dropped_no_motion": dropped_no_motion,
        "dropped_no_card": dropped_no_card,
        "dropped_new_card": dropped_new_card,
    }
    with open(OUTPUT_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSummary: {passed} passed / {total} total")
    print(f"  no_dwpose:  {dropped_no_dwpose}")
    print(f"  no_motion:  {dropped_no_motion}")
    print(f"  no_card:    {dropped_no_card}")
    print(f"  new_card:   {dropped_new_card}")

    # Write annotations
    by_video_pass = defaultdict(list)
    for seg in pass_pool:
        by_video_pass[seg["vname"]].append(seg)

    for vname, segs in by_video_pass.items():
        timeline = []
        for seg in segs:
            card = seg["nearest_card"]
            timeline.append({
                "start_frame": seg["start_frame"],
                "end_frame": seg["end_frame"],
                "labels": ["tap"],
                "meta_text": [],
                "duration_frames": seg["end_frame"] - seg["start_frame"] + 1,
                "id": rand_id(),
                "tap_frame": seg["tap_frame"],
                "tap_point": [seg["tap_x"], seg["tap_y"]],
                "bounding_boxes": [{
                    "frame": seg["card_frame"],
                    "labels": ["card"],
                    "id": rand_id(),
                    "keyframes": [{
                        "frame": seg["card_frame"],
                        "box": card["box"],
                        "polygon_center": card["polygon_center"][0],
                        "rank": card.get("rank", ""),
                        "suit": card.get("suit", ""),
                        "conf": card.get("conf"),
                    }],
                }],
            })
        with open(OUTPUT_DIR / f"{vname}_annotations.json", "w") as f:
            json.dump({
                "video_name": f"{vname}.mp4",
                "video_id": "", "video_path": "",
                "timeline_segments": timeline,
            }, f, indent=2)

    if vis_count != 0:
        print("\nGenerating success vis …")
        visualize_success(pass_pool, vis_count, str(SUCCESS_VIS_DIR))

    if fail_vis_count != 0:
        print("\nGenerating failure vis …")
        visualize_failures(fail_pool, fail_vis_count, str(FAILURE_VIS_DIR))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vis_count", type=int, default=0)
    parser.add_argument("--fail_vis_count", type=int, default=0)
    parser.add_argument("--max_videos", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    process(vis_count=args.vis_count, fail_vis_count=args.fail_vis_count,
            max_videos=args.max_videos, workers=args.workers)
