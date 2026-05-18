"""
Process instance_segments.json to extract verified split segments.

Pipeline:
  Stage 1 – Duration filter   : drop segments < MIN_SEGMENT_FRAMES
  Stage 2 – Same-rank pairing : scan all frames for same-rank card pairs
                                 (K/Q/J/10 grouped) within PAIR_DIST_THRESH (60px)
                                 centroid distance in the player zone (y >= 450px)
  Stage 3 – Movement verify   : track each pair frame-to-frame with suit enforcement;
                                 compute sliding window medians (window=5) of centroid
                                 distance; accept if ceil(max_median - min_median) >= 5px;
                                 among passing pairs, pick the nearest one

Output: annotation JSON per video with 1 bbox + 2 keyframes per split segment.
        Index 0 = upper card at start (smaller y), matched by suit across keyframes.

Usage:
    python process_split_segments.py [--vis_count N] [--fail_vis_count N]
                                     [--max_videos N]
"""

import argparse
import json
import math
import os
import random
import string
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pickle
import numpy as np

# numpy 2.0+ compatibility for loading pickles
if not hasattr(np, '_core'):
    import numpy.core as _np_core
    _sys_modules = __import__('sys').modules
    _sys_modules.setdefault('numpy._core', _np_core)
    for _n in dir(_np_core):
        _sys_modules.setdefault(f'numpy._core.{_n}', getattr(_np_core, _n))

# ─── Paths ───────────────────────────────────────────────────────────────────
SEGMENTS_JSON = (
    "/home/ubuntu/us-west-3-fs/sahithi/hard_action_annotations/"
    "fact_preds_sparse245163_iter26000_cleanhandsplit_allvideos/instance_segments.json"
)
CARDS_DIR = (
    "/home/ubuntu/us-west-3-fs/sahithi/hard_action_annotations/"
    "card_detections_roundwise"
)
RAW_VIDEO_DIR = "/home/ubuntu/us-west-3-fs/live_dealer_blackjack/raw/batch_01"
DWPOSE_DIR = "/home/ubuntu/us-west-3-fs/sahithi/hard_action_annotations/dwpose_roundwise"
OUTPUT_DIR = Path(
    "/home/ubuntu/us-west-3-fs/sahithi/hard_action_annotations/full_run/split_try_2"
)
FAILURE_VIS_DIR = OUTPUT_DIR / "failure_vis"
SUCCESS_VIS_DIR = OUTPUT_DIR / "success_vis"

# ─── Constants ───────────────────────────────────────────────────────────────
IMG_W, IMG_H = 1280, 720
SCALE = 1280 / 1920
MIN_SEGMENT_FRAMES = 25
SPLIT_LABEL = "split"
PAIR_DIST_THRESH = 60       # max centroid dist (px) for a same-rank pair
PLAYER_MIN_Y_PX = 450       # player-zone lower bound
SPREAD_MARGIN = 5           # min (max_dist - min_dist) across all frames
MIN_OVERLAP = 200           # min bbox overlap (px²) to be a valid stacked pair
TOP_K_PAIRS = 10            # max pairs to evaluate in stage 3
FACE_CARD_RANKS = {"K", "Q", "J", "10"}


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def rand_id(n: int = 10) -> str:
    return "".join(random.choices(string.ascii_letters + string.digits, k=n))


def centre_dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _normalize_rank(rank: str) -> str:
    return "10" if rank in FACE_CARD_RANKS else rank


def _scale_det(det: dict) -> dict:
    cx, cy = det["polygon_center"][0]
    x1, y1, x2, y2 = det["box"]
    return {
        **det,
        "polygon_center": [[cx * SCALE, cy * SCALE]],
        "box": [int(x1 * SCALE), int(y1 * SCALE),
                int(x2 * SCALE), int(y2 * SCALE)],
    }


def load_card_detections(jsonl_path: str) -> Dict[int, List[dict]]:
    out: Dict[int, List[dict]] = {}
    with open(jsonl_path) as f:
        for line in f:
            obj = json.loads(line)
            if obj["detections"]:
                out[obj["frame_id"]] = [_scale_det(d) for d in obj["detections"]]
    return out


def nearest_det_frame(
    detections: Dict[int, List[dict]], target: int,
    lo: int, hi: int, prefer: str = "after",
) -> Optional[int]:
    frames = sorted(f for f in detections if lo <= f <= hi)
    if not frames:
        return None
    if prefer == "before":
        cands = [f for f in frames if f <= target]
        return cands[-1] if cands else min(frames, key=lambda f: abs(f - target))
    cands = [f for f in frames if f >= target]
    return cands[0] if cands else min(frames, key=lambda f: abs(f - target))


def player_zone_cards(dets: List[dict]) -> List[dict]:
    return [d for d in dets if d["polygon_center"][0][1] >= PLAYER_MIN_Y_PX]


def _bbox_overlap(a: dict, b: dict) -> int:
    """Pixel area of overlap between two card bounding boxes."""
    ax1, ay1, ax2, ay2 = a["box"]
    bx1, by1, bx2, by2 = b["box"]
    ox = max(0, min(ax2, bx2) - max(ax1, bx1))
    oy = max(0, min(ay2, by2) - max(ay1, by1))
    return ox * oy


def find_same_rank_pairs(
    dets: List[dict], max_dist: float = PAIR_DIST_THRESH,
) -> List[Tuple[dict, dict, float]]:
    """Find same-rank pairs within max_dist that have overlapping bboxes.
    Sorted by overlap (most first), then distance."""
    pairs = []
    for i in range(len(dets)):
        for j in range(i + 1, len(dets)):
            a, b = dets[i], dets[j]
            ra, rb = a.get("rank", ""), b.get("rank", "")
            if not ra or not rb or ra == "U" or rb == "U":
                continue
            if _normalize_rank(ra) != _normalize_rank(rb):
                continue
            d = centre_dist(a["polygon_center"][0], b["polygon_center"][0])
            if d <= max_dist:
                overlap = _bbox_overlap(a, b)
                if overlap >= MIN_OVERLAP:
                    pairs.append((a, b, d, overlap))
    pairs.sort(key=lambda t: (-t[3], t[2]))
    return [(a, b, d) for a, b, d, _ in pairs]


def match_pair_at_frame(
    pair_rank: str, start_centers: Tuple[list, list],
    dets: List[dict], thresh: float = 60,
    start_suits: Tuple[str, str] = ("", ""),
) -> Optional[Tuple[dict, dict, float]]:
    """Match the pair at a new frame. Uses suit to ensure same physical card,
    falls back to proximity if suits are same or unknown."""
    norm = _normalize_rank(pair_rank)
    same = [d for d in dets
            if _normalize_rank(d.get("rank", "")) == norm
            and d.get("rank", "") != "U"]
    if len(same) < 2:
        return None

    c1, c2 = start_centers
    s1, s2 = start_suits

    # if suits differ, use suit to match
    if s1 and s2 and s1 != s2 and s1 != "U" and s2 != "U":
        suit1_cards = [d for d in same if d.get("suit", "") == s1]
        suit2_cards = [d for d in same if d.get("suit", "") == s2]
        if not suit1_cards or not suit2_cards:
            return None
        best_a = min(suit1_cards, key=lambda d: centre_dist(d["polygon_center"][0], c1))
        best_b = min(suit2_cards, key=lambda d: centre_dist(d["polygon_center"][0], c2))
        if centre_dist(best_a["polygon_center"][0], c1) > thresh:
            return None
        if centre_dist(best_b["polygon_center"][0], c2) > thresh:
            return None
    else:
        same.sort(key=lambda d: centre_dist(d["polygon_center"][0], c1))
        best_a = same[0]
        if centre_dist(best_a["polygon_center"][0], c1) > thresh:
            return None
        rest = sorted(
            [d for d in same if d is not best_a],
            key=lambda d: centre_dist(d["polygon_center"][0], c2),
        )
        best_b = rest[0]
        if centre_dist(best_b["polygon_center"][0], c2) > thresh:
            return None

    return (best_a, best_b,
            centre_dist(best_a["polygon_center"][0], best_b["polygon_center"][0]))


def _get_dx(a: dict, b: dict) -> float:
    return abs(a["polygon_center"][0][0] - b["polygon_center"][0][0])


def encompassing_bbox(a: dict, b: dict) -> List[int]:
    ax1, ay1, ax2, ay2 = a["box"]
    bx1, by1, bx2, by2 = b["box"]
    return [min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2)]


def box_to_pct(box: List[int]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return (x1 / IMG_W * 100, y1 / IMG_H * 100,
            (x2 - x1) / IMG_W * 100, (y2 - y1) / IMG_H * 100)


def _base(seg: dict, vname: str) -> dict:
    return {
        "video": vname, "label": SPLIT_LABEL,
        "start_frame": seg["start_frame"], "end_frame": seg["end_frame"],
        "duration_frames": seg["end_frame"] - seg["start_frame"] + 1,
    }


def load_dwpose(pkl_path: str) -> Optional[list]:
    try:
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None


INDEX_TIP_KP = 8
HAND_CONF_THRESH = 0.1


def _get_hand_pos(frame_data: dict, hand_idx: int) -> Optional[Tuple[float, float]]:
    """Get hand position in pixel coords from dwpose. Returns (x, y) or None."""
    hands = np.array(frame_data["pose"]["hands"])  # (2, 21, 2) normalized
    hand = hands[hand_idx]
    if hand.mean() < HAND_CONF_THRESH:
        return None
    for kp in [INDEX_TIP_KP, 0]:  # prefer fingertip, fallback wrist
        x, y = hand[kp]
        if x > 0.01 and y > 0.01:
            return (x * IMG_W, y * IMG_H)
    return None


def _resolve_end_pair(dwpose_data, sf, ef, s0, s1, ea, eb, swapped):
    """Match end cards to start cards. Uses suit → dwpose → swap fallback."""
    s0_suit, s1_suit = s0.get("suit", ""), s1.get("suit", "")

    # different suits → suit match
    if s0_suit and s1_suit and s0_suit != s1_suit and s0_suit != "U" and s1_suit != "U":
        return (ea, eb) if ea.get("suit", "") == s0_suit else (eb, ea)

    # same suit → try dwpose hand tracking
    if dwpose_data and sf < len(dwpose_data) and ef < len(dwpose_data):
        s_h0 = _get_hand_pos(dwpose_data[sf], 0)
        s_h1 = _get_hand_pos(dwpose_data[sf], 1)
        e_h0 = _get_hand_pos(dwpose_data[ef], 0)
        e_h1 = _get_hand_pos(dwpose_data[ef], 1)

        if s_h0 and s_h1 and e_h0 and e_h1:
            s0c = (s0["polygon_center"][0][0], s0["polygon_center"][0][1])
            s1c = (s1["polygon_center"][0][0], s1["polygon_center"][0][1])
            eac = (ea["polygon_center"][0][0], ea["polygon_center"][0][1])
            ebc = (eb["polygon_center"][0][0], eb["polygon_center"][0][1])

            # which hand is near s0 at start?
            if centre_dist(s_h0, s0c) < centre_dist(s_h1, s0c):
                hand_for_s0 = 0
            else:
                hand_for_s0 = 1

            # at end, which card is near that hand?
            end_hand = e_h0 if hand_for_s0 == 0 else e_h1
            if centre_dist(end_hand, eac) < centre_dist(end_hand, ebc):
                return ea, eb
            else:
                return eb, ea

    # fallback: same swap as start
    return (eb, ea) if swapped else (ea, eb)


def make_split_bbox_entry(
    sf: int, sa: dict, sb: dict, ef: int, ea: dict, eb: dict,
    dwpose_data=None,
) -> dict:
    s_box, e_box = encompassing_bbox(sa, sb), encompassing_bbox(ea, eb)
    sx, sy, sw, sh = box_to_pct(s_box)
    ex, ey, ew, eh = box_to_pct(e_box)

    # index 0 = upper card (smaller y) at start
    swapped = sa["polygon_center"][0][1] > sb["polygon_center"][0][1]
    s0, s1 = (sb, sa) if swapped else (sa, sb)
    e0, e1 = _resolve_end_pair(dwpose_data, sf, ef, s0, s1, ea, eb, swapped)

    def _kf(frame, x, y, w, h, bbox, c0, c1):
        return {
            "frame": frame, "x": x, "y": y, "width": w, "height": h,
            "box": bbox,
            "polygon_center": [c0["polygon_center"][0], c1["polygon_center"][0]],
            "card_box": [c0["box"], c1["box"]],
            "rank": [c0.get("rank", ""), c1.get("rank", "")],
            "suit": [c0.get("suit", ""), c1.get("suit", "")],
            "conf": [c0.get("conf"), c1.get("conf")],
        }

    return {
        "frame": sf, "labels": ["card_pair"], "id": rand_id(),
        "keyframes": [
            _kf(sf, sx, sy, sw, sh, s_box, s0, s1),
            _kf(ef, ex, ey, ew, eh, e_box, e0, e1),
        ],
    }


# ═════════════════════════════════════════════════════════════════════════════
# Filtering stages
# ═════════════════════════════════════════════════════════════════════════════

def stage1_duration(segments, vname):
    passed, failed = [], []
    for seg in segments:
        if seg["label"] != SPLIT_LABEL:
            continue
        dur = seg["end_frame"] - seg["start_frame"] + 1
        if dur >= MIN_SEGMENT_FRAMES:
            passed.append(seg)
        else:
            failed.append({**_base(seg, vname), "stage": "stage1_duration",
                           "reason": f"duration={dur} < {MIN_SEGMENT_FRAMES}"})
    return passed, failed


def stage2_same_rank_pair(segments, card_dets, vname):
    if card_dets is None:
        return [], [{**_base(s, vname), "stage": "stage2_no_card_jsonl",
                     "reason": "card jsonl not found"} for s in segments]
    passed, failed = [], []
    for seg in segments:
        sf, ef = seg["start_frame"], seg["end_frame"]

        # Scan all frames for pairs
        seen_pairs = set()
        found_pairs = False
        for f in range(sf, ef + 1):
            if f not in card_dets:
                continue
            dets = player_zone_cards(card_dets[f])
            if len(dets) < 2:
                continue
            pairs = find_same_rank_pairs(dets)
            for ca, cb, dist in pairs:
                norm_rank = _normalize_rank(ca.get("rank", ""))
                # deduplicate by rank + approximate cluster position
                cluster_x = int(min(ca["polygon_center"][0][0], cb["polygon_center"][0][0]) / 50) * 50
                key = (norm_rank, cluster_x)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                passed.append({
                    **seg,
                    "pair_rank": ca.get("rank", ""),
                    "pair_cards_start": (ca, cb),
                    "pair_dist_start": dist,
                    "pair_dx_start": _get_dx(ca, cb),
                    "start_det_frame": f,
                })
                found_pairs = True

        if not found_pairs:
            det_frames = [f for f in range(sf, ef + 1) if f in card_dets]
            if not det_frames:
                failed.append({**_base(seg, vname), "stage": "stage2_no_detections",
                               "reason": "no card detections within segment"})
            else:
                last_f = det_frames[-1]
                n_cards = len(player_zone_cards(card_dets.get(last_f, [])))
                failed.append({**_base(seg, vname), "stage": "stage2_no_same_rank_pair",
                               "reason": f"scanned {len(det_frames)} frames, no same-rank pair found (last: {n_cards} cards at f{last_f})"})
    return passed, failed


WINDOW_SIZE = 5  # sliding window for smoothing distances


def _track_pair(card_dets, rank, start_centers, d_start, dx_start, sf, ef, start_det_f,
                start_suits=("", "")):
    """Track pair across all frames using frame-to-frame matching with suit enforcement.
    Use sliding window medians to find max spread. Robust to noise."""
    measurements = []
    prev_centers = start_centers

    for f in range(start_det_f, ef + 1):
        if f not in card_dets:
            continue
        dets = player_zone_cards(card_dets[f])
        result = match_pair_at_frame(rank, prev_centers, dets,
                                     start_suits=start_suits)
        if result is None:
            continue
        ca_e, cb_e, d_end = result
        prev_centers = (ca_e["polygon_center"][0], cb_e["polygon_center"][0])
        dx_end = _get_dx(ca_e, cb_e)
        measurements.append((f, d_end, ca_e, cb_e, dx_end))

    if len(measurements) < WINDOW_SIZE:
        return -1.0, None

    # compute sliding window medians
    dists = [m[1] for m in measurements]
    w_medians = []
    for i in range(len(dists) - WINDOW_SIZE + 1):
        window = sorted(dists[i:i + WINDOW_SIZE])
        w_medians.append(window[WINDOW_SIZE // 2])

    spread = math.ceil(max(w_medians) - min(w_medians))

    # pick the frame with max distance as the end frame
    best = max(measurements, key=lambda m: m[1])
    best_f, d_end, ca_e, cb_e, dx_end = best
    return spread, (ca_e, cb_e, d_end, dx_end, spread, best_f)


def stage3_spread_verify(segments, card_dets, vname):
    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for seg in segments:
        groups[(seg["start_frame"], seg["end_frame"])].append(seg)

    passed, failed = [], []
    for (sf, ef), candidates in groups.items():
        top = candidates[:TOP_K_PAIRS]

        passing = []
        best_failing, best_fail_move = None, -1.0
        summaries = []

        for seg in top:
            rank = seg["pair_rank"]
            ca_s, cb_s = seg["pair_cards_start"]
            d_start, dx_start = seg["pair_dist_start"], seg["pair_dx_start"]
            start_centers = (ca_s["polygon_center"][0], cb_s["polygon_center"][0])
            start_suits = (ca_s.get("suit", ""), cb_s.get("suit", ""))

            pair_move, pair_best = _track_pair(
                card_dets, rank, start_centers, d_start, dx_start,
                sf, ef, seg["start_det_frame"], start_suits=start_suits,
            )

            if pair_best is None:
                summaries.append(f"{rank}(d={d_start:.0f}):not_found")
                continue

            ca_e, cb_e, d_end, dx_end, dist_range, best_f = pair_best
            summaries.append(
                f"{rank}(d={d_start:.0f}):range={dist_range:.0f} max_d={d_end:.0f} @f{best_f}"
            )

            ok = dist_range >= SPREAD_MARGIN
            if ok:
                passing.append((dist_range, seg, pair_best))
            elif pair_move > best_fail_move:
                best_fail_move = pair_move
                best_failing = (seg, pair_best)

        # Pick passing pair with highest spread
        if passing:
            passing.sort(key=lambda x: -x[0])
            _, best, best_info = passing[0]
            ok = True
        elif best_failing:
            best, best_info = best_failing[0], best_failing[1]
            ok = False
        else:
            best, best_info = None, None
            ok = False

        if ok:
            ca_e, cb_e, d_end, dx_end, _, best_f = best_info
            passed.append({
                **best,
                "pair_cards_end": (ca_e, cb_e),
                "pair_dist_end": d_end,
                "pair_dx_end": dx_end,
                "end_det_frame": best_f,
            })
        else:
            seg0 = top[0]
            info_str = f"best_move={best_fail_move:.0f}px" if best else "no match"
            failed.append({
                **_base(seg0, vname), "stage": "stage3_no_spread",
                "reason": f"{info_str}. top{TOP_K_PAIRS}: [{', '.join(summaries)}]",
                "_best_pair_start": best["pair_cards_start"] if best else None,
                "_best_rank": best["pair_rank"] if best else None,
            })

    return passed, failed


# ═════════════════════════════════════════════════════════════════════════════
# Visualisation
# ═════════════════════════════════════════════════════════════════════════════

def _load_segment_frames(vname: str, sf: int, ef: int):
    try:
        import cv2
    except ImportError:
        return [], 25.0
    parts = vname.split('_')
    base = f"{parts[0]}_{parts[1]}".replace('_', ' ', 1)
    clip_start = int(parts[2])
    for name in (f"{base}.mp4", f"{parts[0]}_{parts[1]}.mp4"):
        path = os.path.join(RAW_VIDEO_DIR, name)
        if os.path.exists(path):
            break
    else:
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


def _write_video(frames, out_path: str, fps: float = 25.0):
    if not frames:
        return
    import cv2
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    h, w = frames[0].shape[:2]
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
    writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    subprocess.run(
        ["/usr/bin/ffmpeg", "-y", "-i", tmp_path,
         "-vcodec", "libx264", "-crf", "23", "-preset", "fast",
         "-movflags", "+faststart", out_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
    )
    os.remove(tmp_path)


def _draw_overlay(frame, cv2, card_dets, fi):
    """Draw player-zone line and top-5 closest same-rank pairs (gray)."""
    cv2.line(frame, (0, PLAYER_MIN_Y_PX), (IMG_W, PLAYER_MIN_Y_PX), (0, 255, 255), 1)
    cv2.putText(frame, "450px", (IMG_W - 60, PLAYER_MIN_Y_PX - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    dets = player_zone_cards(card_dets.get(fi, []))
    for a, b, d in find_same_rank_pairs(dets, PAIR_DIST_THRESH)[:TOP_K_PAIRS]:
        p1 = (int(a["polygon_center"][0][0]), int(a["polygon_center"][0][1]))
        p2 = (int(b["polygon_center"][0][0]), int(b["polygon_center"][0][1]))
        ra, rb = a.get("rank", "?"), b.get("rank", "?")
        gray = (180, 180, 180)
        cv2.line(frame, p1, p2, gray, 1)
        for c in (a, b):
            bx1, by1, bx2, by2 = c["box"]
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), gray, 1)
        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2 - 8)
        cv2.putText(frame, f"{ra}-{rb} d={d:.0f}", mid,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, gray, 1)


def _draw_pair(frame, cv2, ca, cb, color, label):
    """Draw a highlighted card pair with line, boxes, centroids, and label."""
    p1 = (int(ca["polygon_center"][0][0]), int(ca["polygon_center"][0][1]))
    p2 = (int(cb["polygon_center"][0][0]), int(cb["polygon_center"][0][1]))
    for c in (ca, cb):
        bx1, by1, bx2, by2 = c["box"]
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
        cx, cy = int(c["polygon_center"][0][0]), int(c["polygon_center"][0][1])
        cv2.circle(frame, (cx, cy), 5, color, -1)
    cv2.line(frame, p1, p2, color, 2)
    mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2 - 10)
    cv2.putText(frame, label, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def _load_card_dets_for_vis(vname):
    path = os.path.join(CARDS_DIR, f"{vname}_card.jsonl")
    return load_card_detections(path) if os.path.exists(path) else {}


def visualize_success(vis_segments, count, out_dir):
    try:
        import cv2
    except ImportError:
        return
    os.makedirs(out_dir, exist_ok=True)
    pool = list(vis_segments)
    if count != -1 and count < len(pool):
        pool = random.sample(pool, count)

    for idx, info in enumerate(pool, 1):
        vname = info["video_name_base"]
        sf, ef = info["start_frame"], info["end_frame"]
        print(f"  success vis [{idx}/{len(pool)}] {vname} f{sf}-{ef}", flush=True)

        frames, fps = _load_segment_frames(vname, sf, ef)
        if not frames:
            continue

        card_dets = _load_card_dets_for_vis(vname)
        ca_s, cb_s = info["pair_cards_start"]
        rank = info["pair_rank"]
        start_suits = (ca_s.get("suit", ""), cb_s.get("suit", ""))

        # sort start pair: upper=index0 (red), lower=index1 (blue)
        swapped = ca_s["polygon_center"][0][1] > cb_s["polygon_center"][0][1]
        s0, s1 = (cb_s, ca_s) if swapped else (ca_s, cb_s)

        # frame-to-frame tracking for vis
        prev_centers = (s0["polygon_center"][0], s1["polygon_center"][0])
        colors = [(0, 0, 255), (255, 0, 0)]  # red=[0]upper, blue=[1]lower

        for i, frame in enumerate(frames):
            fi = sf + i
            _draw_overlay(frame, cv2, card_dets, fi)

            cur_dets = player_zone_cards(card_dets.get(fi, []))
            live = match_pair_at_frame(rank, prev_centers, cur_dets,
                                       start_suits=start_suits)
            if live:
                c0, c1, ld = live
                prev_centers = (c0["polygon_center"][0], c1["polygon_center"][0])
                for j, c in enumerate([c0, c1]):
                    cx, cy = int(c["polygon_center"][0][0]), int(c["polygon_center"][0][1])
                    bx1, by1, bx2, by2 = c["box"]
                    r, s = c.get("rank", "?"), c.get("suit", "?")
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), colors[j], 2)
                    cv2.circle(frame, (cx, cy), 5, colors[j], -1)
                    cv2.putText(frame, f"[{j}] {r}{s}", (bx1, by1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors[j], 2)

            cv2.putText(
                frame,
                f"SPLIT rank={rank} f{fi} | RED=[0]upper BLUE=[1]lower",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 2,
            )

        _write_video(frames, os.path.join(out_dir, f"split_{vname}_{sf}_{ef}_vis.mp4"), fps)

    print(f"Success vis ({len(pool)} clips) saved under: {out_dir}/")


def visualize_failures(all_failures, count):
    if count == 0:
        return
    try:
        import cv2
    except ImportError:
        return
    pool = list(all_failures)
    if count != -1 and count < len(pool):
        pool = random.sample(pool, count)

    for idx, fc in enumerate(pool, 1):
        stage = fc["stage"]
        vname, sf, ef = fc["video"], fc["start_frame"], fc["end_frame"]
        print(f"  failure vis [{idx}/{len(pool)}] {vname} f{sf}-{ef} ({stage})", flush=True)

        out_path = str(FAILURE_VIS_DIR / stage / f"split_{vname}_{sf}_{ef}.mp4")
        os.makedirs(str(FAILURE_VIS_DIR / stage), exist_ok=True)

        frames, fps = _load_segment_frames(vname, sf, ef)
        if not frames:
            continue

        card_dets = _load_card_dets_for_vis(vname)
        reason = fc.get("reason", "")
        best_start = fc.get("_best_pair_start")
        best_rank = fc.get("_best_rank")
        best_centers = None
        if best_start:
            ca_s, cb_s = best_start
            best_centers = (ca_s["polygon_center"][0], cb_s["polygon_center"][0])

        for i, frame in enumerate(frames):
            fi = sf + i
            cv2.line(frame, (0, PLAYER_MIN_Y_PX), (IMG_W, PLAYER_MIN_Y_PX), (0, 255, 255), 1)

            # draw all same-rank pairs (gray)
            cur_dets = player_zone_cards(card_dets.get(fi, []))
            for a, b, d in find_same_rank_pairs(cur_dets)[:TOP_K_PAIRS]:
                p1 = (int(a["polygon_center"][0][0]), int(a["polygon_center"][0][1]))
                p2 = (int(b["polygon_center"][0][0]), int(b["polygon_center"][0][1]))
                ra = a.get("rank", "?")
                cv2.line(frame, p1, p2, (180, 180, 180), 1)
                for c in (a, b):
                    bx1, by1, bx2, by2 = c["box"]
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (180, 180, 180), 1)
                mid = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2 - 8)
                cv2.putText(frame, f"{ra} d={d:.0f}", mid,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

            # highlight the best candidate pair (orange)
            if best_centers and best_rank:
                live = match_pair_at_frame(best_rank, best_centers, cur_dets)
                if live:
                    la, lb, ld = live
                    _draw_pair(frame, cv2, la, lb, (0, 165, 255),
                               f"d={ld:.0f}")

            cv2.putText(frame, f"FAIL [{stage}] f{fi}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            cv2.putText(frame, reason[:140],
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        _write_video(frames, out_path, fps)

    print(f"Failure vis ({len(pool)} clips) saved under: {FAILURE_VIS_DIR}/")


# ═════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═════════════════════════════════════════════════════════════════════════════

def _write_annotation(vname, video_name, timeline_segments):
    record = {
        "video_name": video_name, "video_id": "", "video_path": "",
        "timeline_segments": timeline_segments,
    }
    with open(OUTPUT_DIR / f"{vname}_annotations.json", "w") as f:
        json.dump(record, f, indent=2)


def process(vis_count=0, fail_vis_count=0, max_videos=0):
    print(f"Loading instance segments from: {SEGMENTS_JSON}")
    with open(SEGMENTS_JSON) as f:
        all_segments: Dict[str, List[dict]] = json.load(f)

    split_videos = {
        k: v for k, v in all_segments.items()
        if any(s["label"] == SPLIT_LABEL for s in v)
    }
    n_split = sum(1 for v in split_videos.values()
                  for s in v if s["label"] == SPLIT_LABEL)
    print(f"Found {n_split} split segments across {len(split_videos)} videos")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = {p.stem.replace("_annotations", "")
                for p in OUTPUT_DIR.glob("*_annotations.json")}

    all_failures: List[dict] = []
    vis_segments: List[dict] = []
    keys = list(split_videos.keys())[:max_videos] if max_videos else list(split_videos.keys())
    n_skipped = 0

    for vi, vname in enumerate(keys, 1):
        segments = split_videos[vname]
        video_name = vname + ".mp4"

        if vname in existing:
            n_skipped += 1
            continue

        # Stage 1
        s1_pass, s1_fail = stage1_duration(segments, vname)
        all_failures.extend(s1_fail)

        if not s1_pass:
            _write_annotation(vname, video_name, [])
            continue

        # Load card detections and dwpose
        jsonl = os.path.join(CARDS_DIR, f"{vname}_card.jsonl")
        card_dets = load_card_detections(jsonl) if os.path.exists(jsonl) else None
        dwpose_path = os.path.join(DWPOSE_DIR, f"{vname}_dwpose.pkl")
        dwpose_data = load_dwpose(dwpose_path) if os.path.exists(dwpose_path) else None

        if vi % 200 == 0 or len(s1_pass) > 1:
            print(f"[{vi}/{len(keys)}] {vname}  ({len(s1_pass)} split segs) …",
                  flush=True)

        # Stage 2
        s2_pass, s2_fail = stage2_same_rank_pair(s1_pass, card_dets, vname)
        all_failures.extend(s2_fail)

        # Stage 3
        if card_dets is not None:
            s3_pass, s3_fail = stage3_spread_verify(s2_pass, card_dets, vname)
        else:
            s3_pass, s3_fail = [], []
        all_failures.extend(s3_fail)

        # Build annotation
        success_map = {(s["start_frame"], s["end_frame"]): s for s in s3_pass}
        timeline = []
        for seg in segments:
            if seg["label"] != SPLIT_LABEL:
                continue
            sf, ef = seg["start_frame"], seg["end_frame"]
            dur = ef - sf + 1
            if dur < MIN_SEGMENT_FRAMES:
                continue
            bboxes = []
            if (sf, ef) in success_map:
                s = success_map[(sf, ef)]
                bboxes = [make_split_bbox_entry(
                    s["start_det_frame"], *s["pair_cards_start"],
                    s["end_det_frame"], *s["pair_cards_end"],
                    dwpose_data=dwpose_data,
                )]
            timeline.append({
                "start_frame": sf, "end_frame": ef,
                "labels": [SPLIT_LABEL], "meta_text": [],
                "duration_frames": dur, "id": rand_id(),
                "bounding_boxes": bboxes,
            })
        _write_annotation(vname, video_name, timeline)

        for s in s3_pass:
            vis_segments.append({
                "video_name_base": vname, "label": SPLIT_LABEL,
                "start_frame": s["start_frame"], "end_frame": s["end_frame"],
                "pair_rank": s["pair_rank"],
                "pair_cards_start": s["pair_cards_start"],
                "pair_cards_end": s["pair_cards_end"],
                "pair_dist_start": s["pair_dist_start"],
                "pair_dist_end": s["pair_dist_end"],
                "pair_dx_start": s["pair_dx_start"],
                "pair_dx_end": s["pair_dx_end"],
                "start_det_frame": s["start_det_frame"],
                "end_det_frame": s["end_det_frame"],
            })

    # Stats & output
    stage_counts: Dict[str, int] = {}
    for fc in all_failures:
        stage_counts[fc["stage"]] = stage_counts.get(fc["stage"], 0) + 1

    total = len(vis_segments) + len(all_failures)
    with open(OUTPUT_DIR / "stats.json", "w") as f:
        json.dump({
            "total_split_segments": total,
            "successful": len(vis_segments),
            "failed_total": len(all_failures),
            "failed_by_stage": stage_counts,
            "skipped_resume": n_skipped,
        }, f, indent=2)

    clean = lambda fc: {k: v for k, v in fc.items() if not k.startswith("_")}
    with open(OUTPUT_DIR / "failure_cases.json", "w") as f:
        json.dump([clean(fc) for fc in all_failures], f, indent=2)

    print(f"\nSummary: {len(vis_segments)} successful  /  "
          f"{len(all_failures)} failed  /  {total} total"
          f"  ({n_skipped} skipped from previous run)")
    for stage, cnt in sorted(stage_counts.items()):
        print(f"  {stage}: {cnt}")
    print(f"Stats   : {OUTPUT_DIR / 'stats.json'}")
    print(f"Failures: {OUTPUT_DIR / 'failure_cases.json'}")

    if fail_vis_count != 0:
        n = len(all_failures) if fail_vis_count == -1 else fail_vis_count
        print(f"\nGenerating {n} failure visualisations …")
        visualize_failures(all_failures, fail_vis_count)

    if vis_count != 0:
        n = len(vis_segments) if vis_count == -1 else vis_count
        print(f"\nGenerating {n} success visualisations …")
        visualize_success(vis_segments, vis_count, str(SUCCESS_VIS_DIR))

    return vis_segments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vis_count", type=int, default=0,
                        help="Success clips: 0=none, N=random N, -1=all")
    parser.add_argument("--fail_vis_count", type=int, default=0,
                        help="Failure clips: 0=none, N=random N, -1=all")
    parser.add_argument("--max_videos", type=int, default=0,
                        help="Limit to first N videos (0=all)")
    args = parser.parse_args()
    process(vis_count=args.vis_count, fail_vis_count=args.fail_vis_count,
            max_videos=args.max_videos)
