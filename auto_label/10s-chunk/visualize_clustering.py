"""
Visualize card clustering at the exact frame used for meta_text computation.

For each selected prediction JSON the script:
  1. Finds the "initial hands 1st" segment
  2. Looks up card detections at its last frame (the frame used for clustering)
  3. Runs clustering and draws the result on a 1920x1080 canvas
  4. Saves an annotated PNG to --output-dir

File selection
--------------
  --filename BASENAME    Visualize a specific prediction JSON by basename (repeatable)
  --n N                  Visualize N files total (random by default, or first N with
                         --no-random).  If --filename is also given, those files are
                         included and the remaining slots are filled randomly.
                         Default: --n 5 when no --filename is given.

Examples
--------
  # 5 random files
  python visualize_clustering.py

  # first 10 files (alphabetical order)
  python visualize_clustering.py --n 10 --no-random

  # one specific file
  python visualize_clustering.py --filename 2024-01-15_12-30_001234_001534_predictions.json

  # specific file + 4 random others
  python visualize_clustering.py --filename foo.json --n 5
"""

import argparse
import glob
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from clustering_util import (
    IMAGE_H,
    IMAGE_W,
    MANUAL_TEMPLATE_NORM,
    N_POSITIONS,
    assign_cards_to_positions,
    get_active_players,
)


# ==================== DEFAULT PATHS ====================

PRED_JSON_DIR = (
    "/home/ubuntu/yifan/code/motion-data-process/auto_label/10s-chunk"
    "/data/inference_results_filtered_videos_26k/predictions_json"
)
CARD_DETECTION_DIR = (
    "/home/ubuntu/us-west-3-fs/live_dealer_blackjack/yolo_cards/card_detection/all_jsons"
)
OUTPUT_DIR = (
    "/home/ubuntu/yifan/code/motion-data-process/auto_label/10s-chunk"
    "/data/inference_results_filtered_videos_26k/clustering_vis"
)

FRAMES_PER_CARD_FILE = 300


# ==================== COLOUR PALETTE ====================

COLOUR_DEALER   = (180, 180,   0)   # yellow-ish (BGR)
COLOUR_ACTIVE   = (  0, 200,  50)   # green
COLOUR_INACTIVE = ( 80,  80,  80)   # dark grey
COLOUR_CARD     = (  0, 100, 255)   # orange-red
COLOUR_LINE     = (200, 200, 200)   # light grey
COLOUR_TEXT     = (255, 255, 255)   # white
COLOUR_BG_TEXT  = (  0,   0,   0)   # black

SEAT_LABELS = ["D", "P1", "P2", "P3", "P4", "P5", "P6", "P7"]


# ==================== DRAWING HELPERS ====================

def _template_pixels(image_w: int = IMAGE_W, image_h: int = IMAGE_H) -> np.ndarray:
    scale = np.array([image_w, image_h], dtype=np.float64)
    return (MANUAL_TEMPLATE_NORM * scale).astype(int)


def _draw_seat(canvas, center, label, colour, radius=22):
    cx, cy = int(center[0]), int(center[1])
    cv2.circle(canvas, (cx, cy), radius, colour, thickness=2)
    cv2.circle(canvas, (cx, cy), 3, colour, thickness=-1)
    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    tx = cx - text_size[0] // 2
    ty = cy - radius - 6
    cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)


def _draw_card(canvas, center, colour=COLOUR_CARD):
    cx, cy = int(center[0]), int(center[1])
    cv2.drawMarker(canvas, (cx, cy), colour, cv2.MARKER_TILTED_CROSS, 16, 2, cv2.LINE_AA)


def _put_text_with_bg(canvas, text, origin, font_scale=0.7, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = origin
    pad = 4
    cv2.rectangle(canvas, (x - pad, y - th - pad), (x + tw + pad, y + baseline + pad),
                  COLOUR_BG_TEXT, -1)
    cv2.putText(canvas, text, (x, y), font, font_scale, COLOUR_TEXT, thickness, cv2.LINE_AA)


# ==================== CORE VISUALIZER ====================

def visualize_clustering(
    detections: List[Dict],
    background: Optional[np.ndarray] = None,
    image_w: int = IMAGE_W,
    image_h: int = IMAGE_H,
    title: str = "",
) -> np.ndarray:
    """
    Draw clustering result on a canvas and return the annotated image.

    Args:
        detections:       Detection dicts with 'polygon_center' in pixel coords.
        background:       Optional BGR image; if None a dark blank canvas is created.
        image_w, image_h: Canvas / frame dimensions.
        title:            Optional string shown at the bottom of the image.
    """
    if background is not None:
        canvas = cv2.resize(background.copy(), (image_w, image_h))
    else:
        canvas = np.full((image_h, image_w, 3), 30, dtype=np.uint8)

    template_px = _template_pixels(image_w, image_h)
    clusters: Dict[int, list] = assign_cards_to_positions(
        detections, image_w=image_w, image_h=image_h
    )
    active_str = get_active_players(detections)

    # Assignment lines (drawn first, underneath everything)
    for pos_idx, cards in clusters.items():
        seat_pt = tuple(template_px[pos_idx])
        for det in cards:
            cx, cy = int(det["polygon_center"][0]), int(det["polygon_center"][1])
            cv2.line(canvas, (cx, cy), seat_pt, COLOUR_LINE, 1, cv2.LINE_AA)

    # Seat circles
    for pos_idx in range(N_POSITIONS):
        seat_pt = tuple(template_px[pos_idx])
        if pos_idx == 0:
            colour = COLOUR_DEALER
        elif clusters[pos_idx]:
            colour = COLOUR_ACTIVE
        else:
            colour = COLOUR_INACTIVE
        _draw_seat(canvas, seat_pt, SEAT_LABELS[pos_idx], colour)

    # Card centroids
    for det in detections:
        _draw_card(canvas, det["polygon_center"])

    # HUD — top-left
    y = 34
    _put_text_with_bg(canvas, f"Active players: {active_str or '(none)'}", (12, y))
    y += 36
    _put_text_with_bg(canvas, f"Cards detected: {len(detections)}", (12, y))

    assigned = sum(len(v) for v in clusters.values())
    unassigned = len(detections) - assigned
    if unassigned > 0:
        y += 36
        _put_text_with_bg(canvas, f"Unassigned: {unassigned}", (12, y))

    # Title — bottom
    if title:
        _put_text_with_bg(canvas, title, (12, image_h - 12), font_scale=0.55, thickness=1)

    return canvas


# ==================== FILE-PARSING & CARD-LOOKUP ====================

def _parse_pred_filename(path: str) -> Tuple[str, int, int]:
    """Returns (base_name, clip_start_1idx, clip_end_1idx)."""
    base = os.path.basename(path).replace("_predictions.json", "")
    parts = base.split("_")
    base_name = f"{parts[0]}_{parts[1]}"
    return base_name, int(parts[2]), int(parts[3])


def _parse_card_filename(path: str) -> Tuple[str, int, int]:
    base = os.path.basename(path).replace("_card.json", "")
    parts = base.split("_")
    base_name = f"{parts[0]}_{parts[1]}"
    return base_name, int(parts[2]), int(parts[3])


def _group_card_files(card_dir: str) -> Dict[str, List[Tuple[str, int, int]]]:
    grouped: Dict[str, List] = defaultdict(list)
    for fp in glob.glob(os.path.join(card_dir, "*_card.json")):
        try:
            base_name, _, start_frame = _parse_card_filename(fp)
            grouped[base_name].append((fp, start_frame))
        except Exception:
            continue
    for k in grouped:
        grouped[k].sort(key=lambda x: x[1])
    return grouped


def _find_card_files_for_range(
    card_group: List[Tuple[str, int]], global_start: int, global_end: int
) -> List[Tuple[str, int, int]]:
    results = []
    for fp, card_start in card_group:
        card_end = card_start + FRAMES_PER_CARD_FILE - 1
        if card_end >= global_start and card_start <= global_end:
            results.append((fp, card_start, card_end))
    return results


def _load_detections_at_frame(
    card_files: List[Tuple[str, int, int]], target_frame: int
) -> List[Dict]:
    for fp, start_frame, end_frame in card_files:
        if start_frame <= target_frame <= end_frame:
            with open(fp) as f:
                frame_data = json.load(f)
            rel = target_frame - start_frame
            for entry in frame_data:
                if isinstance(entry, dict) and entry.get("frame_id") == rel:
                    return entry.get("detections", [])
            if 0 <= rel < len(frame_data):
                return frame_data[rel].get("detections", [])
    return []


def _get_clustering_frame_info(
    pred_path: str, card_grouped: Dict
) -> Optional[Tuple[int, List[Dict]]]:
    """
    Return (global_frame, detections) for the last frame of "initial hands 1st",
    or None if the segment is missing or no card files are found.
    """
    base_name, clip_start_1idx, _ = _parse_pred_filename(pred_path)
    with open(pred_path) as f:
        pred_data = json.load(f)

    for seg in pred_data.get("timeline_segments", []):
        if "initial hands 1st" in seg.get("labels", []):
            global_end = (clip_start_1idx - 1) + seg["end_frame"]
            global_start = (clip_start_1idx - 1) + seg["start_frame"]
            card_group = card_grouped.get(base_name, [])
            card_files = _find_card_files_for_range(card_group, global_start, global_end)
            if not card_files:
                return None
            detections = _load_detections_at_frame(card_files, global_end)
            return global_end, detections
    return None


# ==================== CLI ====================

def _parse_args():
    p = argparse.ArgumentParser(
        description="Visualize card clustering at the frame used for meta_text computation."
    )
    p.add_argument(
        "--filename", "-f",
        action="append",
        default=[],
        metavar="BASENAME",
        help="Prediction JSON basename (or full path) to visualize.  Repeatable.",
    )
    p.add_argument(
        "--n",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Total number of files to visualize.  Named files (--filename) are included "
            "first; remaining slots are filled randomly (or by sort order with --no-random). "
            "Default: 5 when no --filename is given."
        ),
    )
    p.add_argument(
        "--no-random",
        action="store_true",
        help="Fill remaining slots in alphabetical order instead of randomly.",
    )
    p.add_argument(
        "--pred-dir",
        default=PRED_JSON_DIR,
        metavar="DIR",
        help="Directory containing *_predictions.json files.",
    )
    p.add_argument(
        "--card-dir",
        default=CARD_DETECTION_DIR,
        metavar="DIR",
        help="Directory containing *_card.json files.",
    )
    p.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        metavar="DIR",
        help="Directory where PNG visualizations are saved.",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Collect all available prediction files ----
    all_pred = sorted(glob.glob(os.path.join(args.pred_dir, "*_predictions.json")))
    if not all_pred:
        print(f"No prediction JSON files found in: {args.pred_dir}")
        return

    basename_to_path = {os.path.basename(p): p for p in all_pred}

    # ---- Resolve named files ----
    named_paths = []
    for spec in args.filename:
        key = os.path.basename(spec)
        if key in basename_to_path:
            named_paths.append(basename_to_path[key])
        elif os.path.isfile(spec):
            named_paths.append(spec)
        else:
            print(f"Warning: --filename '{spec}' not found, skipping.")

    # ---- Fill remaining slots ----
    n_total = args.n if args.n is not None else (5 if not named_paths else len(named_paths))
    n_extra = max(0, n_total - len(named_paths))
    pool = [p for p in all_pred if p not in named_paths]
    if n_extra > 0:
        if args.no_random:
            extra = pool[:n_extra]
        else:
            extra = random.sample(pool, min(n_extra, len(pool)))
    else:
        extra = []

    selected = named_paths + extra
    print(f"Visualizing {len(selected)} file(s)  →  {args.output_dir}")

    # ---- Load card detection index ----
    print("Indexing card detection files …")
    card_grouped = _group_card_files(args.card_dir)

    # ---- Process each file ----
    saved = 0
    for pred_path in selected:
        try:
            result = _get_clustering_frame_info(pred_path, card_grouped)
        except Exception as e:
            print(f"  ERROR  {os.path.basename(pred_path)}: {e}")
            continue

        if result is None:
            print(f"  SKIP   {os.path.basename(pred_path)}  (no 'initial hands 1st' or no card data)")
            continue

        global_frame, detections = result
        active_str = get_active_players(detections)

        title = (
            f"{os.path.basename(pred_path)}  |  frame {global_frame}  |  "
            f"cards: {len(detections)}  |  active: {active_str or 'none'}"
        )
        canvas = visualize_clustering(detections, title=title)

        stem = os.path.basename(pred_path).replace("_predictions.json", "")
        out_path = os.path.join(args.output_dir, f"{stem}_frame{global_frame}_clustering.png")
        cv2.imwrite(out_path, canvas)
        print(f"  SAVED  {os.path.basename(out_path)}  active={active_str or 'none'}  cards={len(detections)}")
        saved += 1

    print(f"\nDone — {saved}/{len(selected)} visualization(s) saved.")


if __name__ == "__main__":
    main()
