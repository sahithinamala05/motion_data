#!/usr/bin/env python3
"""
Compute meta_text for prediction JSON files and save updated copies.

For each prediction JSON:
1. Find the "initial hands 1st" segment
2. Look up the card detection result at its last frame
3. Cluster cards to determine active player seats (1-7)
4. Write meta_text (e.g. "12357") into "initial hands 1st", "initial hands 2nd", "discard" segments
5. Save the updated JSON to the output directory
"""

import os
import json
import glob
from collections import defaultdict
from typing import Dict, List, Tuple

from tqdm import tqdm

from clustering_util import get_active_players


# ==================== CONFIGURATION ====================
PRED_JSON_DIR = "/home/ubuntu/yifan/code/motion-data-process/auto_label/10s-chunk/data/inference_results_filtered_videos_26k/predictions_json"
CARD_DETECTION_DIR = "/home/ubuntu/us-west-3-fs/live_dealer_blackjack/yolo_cards/card_detection/all_jsons"
OUTPUT_DIR = "/home/ubuntu/yifan/code/motion-data-process/auto_label/10s-chunk/data/inference_results_filtered_videos_26k/predictions_json_with_meta"

FRAMES_PER_CARD_FILE = 300

META_TEXT_LABELS = {'initial hands 1st', 'initial hands 2nd', 'discard'}


# ==================== FILE PARSING ====================

def parse_pred_json_filename(filename: str) -> Tuple[str, int, int]:
    """Parse YYYY-MM-DD_HH-MM-SS_XXXXXX_YYYYYY_predictions.json (1-indexed frames)."""
    basename = os.path.basename(filename).replace('_predictions.json', '')
    parts = basename.split('_')
    base_name = f"{parts[0]}_{parts[1]}"
    return base_name, int(parts[2]), int(parts[3])


def parse_card_json_filename(filename: str) -> Tuple[str, int, int]:
    """Parse YYYY-MM-DD_HH-MM-SS_XXXXXX_YYYYYY_card.json (0-indexed clip/frame)."""
    basename = os.path.basename(filename).replace('_card.json', '')
    parts = basename.split('_')
    base_name = f"{parts[0]}_{parts[1]}"
    return base_name, int(parts[2]), int(parts[3])


def group_card_detection_files(card_dir: str) -> Dict[str, List[Tuple[str, int, int]]]:
    """Group card detection files by YYYY-MM-DD_HH-MM-SS."""
    grouped = defaultdict(list)
    for filepath in glob.glob(os.path.join(card_dir, "*_card.json")):
        try:
            base_name, clip_index, start_frame = parse_card_json_filename(filepath)
            grouped[base_name].append((filepath, clip_index, start_frame))
        except Exception:
            continue
    for base_name in grouped:
        grouped[base_name].sort(key=lambda x: x[2])
    print(f"Card detection: {sum(len(v) for v in grouped.values())} files across {len(grouped)} videos")
    return grouped


# ==================== CARD DETECTION LOOKUP ====================

def find_card_files_for_frame_range(
    card_group: List[Tuple[str, int, int]],
    global_start: int,
    global_end: int,
) -> List[Tuple[str, int, int]]:
    """Find card detection files overlapping [global_start, global_end]."""
    results = []
    for filepath, clip_index, card_start in card_group:
        card_end = card_start + FRAMES_PER_CARD_FILE - 1
        if card_end >= global_start and card_start <= global_end:
            results.append((filepath, card_start, card_end))
    return results


def load_card_detections_for_frame(
    card_files: List[Tuple[str, int, int]],
    target_frame: int,
) -> List[Dict]:
    """Load card detections for a specific global frame."""
    for filepath, start_frame, end_frame in card_files:
        if start_frame <= target_frame <= end_frame:
            with open(filepath, 'r') as f:
                frame_data = json.load(f)
            relative_frame = target_frame - start_frame
            for entry in frame_data:
                if entry.get('frame_id') == relative_frame:
                    return entry.get('detections', [])
            if 0 <= relative_frame < len(frame_data):
                return frame_data[relative_frame].get('detections', [])
    return []


# ==================== CORE LOGIC ====================

def compute_meta_text_for_prediction(
    pred_data: dict,
    clip_start_1idx: int,
    card_files: List[Tuple[str, int, int]],
) -> str:
    """Find 'initial hands 1st' segment, cluster last frame, return meta_text."""
    for segment in pred_data.get('timeline_segments', []):
        if 'initial hands 1st' in segment.get('labels', []):
            global_start = (clip_start_1idx - 1) + segment['start_frame']
            global_end = (clip_start_1idx - 1) + segment['end_frame']

            matching = find_card_files_for_frame_range(card_files, global_start, global_end)
            if not matching:
                return ""

            detections = load_card_detections_for_frame(matching, global_end)
            return get_active_players(detections)
    return ""


# ==================== MAIN ====================

def main():
    print("=" * 70)
    print("COMPUTE META_TEXT FOR PREDICTION JSON FILES (v2)")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Group card detection files
    card_grouped = group_card_detection_files(CARD_DETECTION_DIR)

    # Find prediction files
    pred_files = sorted(glob.glob(os.path.join(PRED_JSON_DIR, "*_predictions.json")))
    print(f"Prediction JSON files: {len(pred_files)}")

    # Stats
    total_files = 0
    files_with_meta = 0
    label_counts = defaultdict(int)

    for pred_path in tqdm(pred_files, desc="Processing"):
        try:
            base_name, clip_start_1idx, clip_end_1idx = parse_pred_json_filename(pred_path)
        except Exception as e:
            print(f"\nSkipping {pred_path}: {e}")
            continue

        with open(pred_path, 'r') as f:
            pred_data = json.load(f)

        total_files += 1

        # Compute meta_text from card clustering
        card_files = card_grouped.get(base_name, [])
        meta_text = compute_meta_text_for_prediction(
            pred_data, clip_start_1idx, card_files
        )

        # Apply meta_text to matching segments
        updated = False
        for segment in pred_data.get('timeline_segments', []):
            label = segment['labels'][0] if segment.get('labels') else ''
            if label in META_TEXT_LABELS:
                segment['meta_text'] = [meta_text]
                label_counts[label] += 1
                if meta_text:
                    updated = True

        if updated:
            files_with_meta += 1

        # Save updated JSON
        out_path = os.path.join(OUTPUT_DIR, os.path.basename(pred_path))
        with open(out_path, 'w') as f:
            json.dump(pred_data, f, indent=2)

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total files processed: {total_files}")
    print(f"Files with non-empty meta_text: {files_with_meta}")
    print(f"\nSegments updated per label:")
    for label in sorted(label_counts.keys()):
        print(f"  {label}: {label_counts[label]}")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
