#!/usr/bin/env python3
"""
Merge per-300-frame card detection JSON files into per-round JSONL files.

Follows the same pattern as form_action_data_roundwise.py but for card detections.

For each round video (YYYY-MM-DD_HH-MM-SS_XXXXXX_YYYYYY):
1. Find all 300-frame card detection JSONs that overlap with the round's global frame range
2. Concatenate them in order, trimming to round boundaries
3. Normalize field names to match the round-level format expected by process_hit_segments.py
4. Save as one JSONL file (one JSON line per frame)
5. Optionally visualize with card bounding boxes on source video

Usage:
    # Examples-only mode (process + visualize a few clips first):
    python merge_cards_roundwise.py --examples-only 3

    # Full processing (no visualization):
    python merge_cards_roundwise.py

    # Full processing + visualize N random examples:
    python merge_cards_roundwise.py --vis-examples 5
"""

import json
import os
import glob
import subprocess
import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

# ─── paths ────────────────────────────────────────────────────────────────────
CARD_CHUNK_DIR = (
    "/home/ubuntu/us-west-3-fs/live_dealer_blackjack/"
    "yolo_cards/card_detection_n_classification/all_jsons"
)
# Round names come from the FACT predictions
PREDICTIONS_DIR = (
    "/home/ubuntu/us-west-3-fs/sahithi/hard_action_annotations/predictions_json"
)
OUTPUT_DIR = (
    "/home/ubuntu/us-west-3-fs/sahithi/hard_action_annotations/card_detections_roundwise"
)
VIDEO_DIR = "/home/ubuntu/us-west-3-fs/live_dealer_blackjack/raw/batch_01"
VIS_DIR = "/home/ubuntu/sahithi/vis/card_vis"

FRAMES_PER_CHUNK = 300

# Colors per suit (BGR for cv2)
SUIT_COLORS = {
    'H': (0, 0, 255),     # red
    'D': (0, 100, 255),   # orange-red
    'S': (255, 200, 100), # light blue
    'C': (100, 255, 100), # green
}


# ─── filename parsing ─────────────────────────────────────────────────────────

def parse_chunk_filename(filename: str) -> Tuple[str, int, int]:
    """
    Parse card detection chunk filename.
    Format: YYYY-MM-DD_HH-MM-SS_XXXXXX_YYYYYY_card.json
    Returns: (base_name, chunk_index, start_frame_0indexed)
    """
    basename = os.path.basename(filename).replace('_card.json', '')
    parts = basename.split('_')
    if len(parts) != 4:
        raise ValueError(f"Invalid chunk filename: {filename}")
    base_name = f"{parts[0]}_{parts[1]}"
    chunk_index = int(parts[2])
    start_frame = int(parts[3])
    return base_name, chunk_index, start_frame


def parse_round_name(name: str) -> Tuple[str, int, int]:
    """
    Parse round name from prediction JSON filename.
    Format: YYYY-MM-DD_HH-MM-SS_XXXXXX_YYYYYY.json
    XXXXXX/YYYYYY are 1-indexed.
    Returns: (base_name, round_start_1indexed, round_end_1indexed)
    """
    basename = os.path.basename(name).replace('.json', '')
    parts = basename.split('_')
    if len(parts) != 4:
        raise ValueError(f"Invalid round name: {name}")
    base_name = f"{parts[0]}_{parts[1]}"
    round_start = int(parts[2])  # 1-indexed
    round_end = int(parts[3])    # 1-indexed
    return base_name, round_start, round_end


def find_source_video(base_name: str) -> Optional[str]:
    """Find the source video for a base_name like '2025-10-01_06-05-15'."""
    video_filename = base_name.replace('_', ' ', 1) + ".mp4"
    video_path = os.path.join(VIDEO_DIR, video_filename)
    if os.path.exists(video_path):
        return video_path
    return None


# ─── group chunks by base video ──────────────────────────────────────────────

def group_chunk_files(chunk_dir: str) -> Dict[str, List[Tuple[str, int, int]]]:
    """Returns: base_name -> sorted list of (filepath, chunk_index, start_frame_0indexed)"""
    grouped = defaultdict(list)
    chunk_files = glob.glob(os.path.join(chunk_dir, "*_card.json"))
    print(f"Found {len(chunk_files)} card chunk files in {chunk_dir}")

    for filepath in tqdm(chunk_files, desc="Parsing chunk filenames"):
        try:
            base_name, chunk_index, start_frame = parse_chunk_filename(filepath)
            grouped[base_name].append((filepath, chunk_index, start_frame))
        except Exception:
            pass

    for base_name in grouped:
        grouped[base_name].sort(key=lambda x: x[2])

    print(f"Grouped into {len(grouped)} base videos")
    return grouped


# ─── normalize detection format ──────────────────────────────────────────────

def normalize_detection(det: dict) -> dict:
    """Convert chunk format to round-level format expected by process_hit_segments.py."""
    pc = det.get("polygon_center", [0, 0])
    # Ensure polygon_center is [[x, y]] not [x, y]
    if pc and not isinstance(pc[0], list):
        pc = [pc]

    bbox = det.get("bbox", det.get("box", [0, 0, 0, 0]))

    return {
        "polygon_center": pc,
        "box": [int(round(v)) for v in bbox],
        "conf": det.get("confidence", det.get("conf", 0)),
        "rank": det.get("rank", "U"),
        "rank_conf": det.get("rank_conf", det.get("confidence", 0)),
        "suit": det.get("suit", "U"),
        "suit_conf": det.get("suit_conf", det.get("confidence", 0)),
        "match_type": det.get("match_type", "rank+suit" if det.get("rank") else "unknown"),
    }


# ─── merge logic ─────────────────────────────────────────────────────────────

def merge_chunks_for_round(
    chunk_group: List[Tuple[str, int, int]],
    global_start_0idx: int,
    global_end_0idx: int,
) -> Optional[List[dict]]:
    """
    Merge card detection chunks for a round's global frame range.
    Returns list of {frame_id, detections} dicts, or None if no overlap.
    """
    matching = []
    for filepath, chunk_index, chunk_start in chunk_group:
        chunk_end = chunk_start + FRAMES_PER_CHUNK - 1
        if chunk_end >= global_start_0idx and chunk_start <= global_end_0idx:
            matching.append((filepath, chunk_start, chunk_end))

    if not matching:
        return None

    matching.sort(key=lambda x: x[1])

    merged = []
    for filepath, chunk_start, chunk_end in matching:
        with open(filepath) as f:
            frames = json.load(f)

        slice_start = max(0, global_start_0idx - chunk_start)
        slice_end = min(len(frames), global_end_0idx - chunk_start + 1)

        for frame_data in frames[slice_start:slice_end]:
            local_frame_id = len(merged)
            normalized_dets = [normalize_detection(d) for d in frame_data.get("detections", [])]
            merged.append({
                "frame_id": local_frame_id,
                "detections": normalized_dets,
            })

    return merged


# ─── visualization ───────────────────────────────────────────────────────────

def get_label_at_frame(timeline_segments: list, frame_idx: int) -> str:
    """Return the action label for the given relative frame index, or '' if none."""
    for seg in timeline_segments:
        if seg['start_frame'] <= frame_idx <= seg['end_frame']:
            labels = seg.get('labels', [])
            return labels[0] if labels else ''
    return ''


def draw_card_detections(frame, detections):
    """Draw card detection bounding boxes and labels on a video frame (BGR)."""
    import cv2

    for det in detections:
        bbox = det.get("box", det.get("bbox", []))
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        rank = det.get("rank", "?")
        suit = det.get("suit", "?")
        conf = det.get("conf", det.get("confidence", 0))
        color = SUIT_COLORS.get(suit, (200, 200, 200))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label_text = f"{rank}{suit} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label_text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    return frame


def visualize_merged_round(
    merged_card_data: List[dict],
    json_path: str,
    vis_dir: str,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
):
    """
    Visualize a merged round with card detection boxes on the source video,
    plus action label overlay from the prediction JSON.
    """
    import cv2

    # Load annotation for action label overlay
    with open(json_path, 'r') as f:
        annotation = json.load(f)
    timeline_segments = annotation.get('timeline_segments', [])

    round_stem = os.path.basename(json_path).replace('.json', '')
    total = len(merged_card_data)
    print(f"  Visualizing {round_stem}: {total} frames")

    # Find source video and seek
    base_name, round_start_1idx, round_end_1idx = parse_round_name(json_path)
    global_start_frame = round_start_1idx - 1

    video_path = find_source_video(base_name)
    cap = None
    if video_path:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_FRAMES, global_start_frame)
            print(f"  Source video: {os.path.basename(video_path)} (seeking to frame {global_start_frame})")
        else:
            print(f"  Warning: Could not open source video, falling back to black background")
            cap.release()
            cap = None
    else:
        print(f"  Warning: Source video not found for {base_name}, falling back to black background")

    # Video output
    temp_path = os.path.join(vis_dir, f"{round_stem}_card_vis_temp.mp4")
    final_path = os.path.join(vis_dir, f"{round_stem}_card_vis.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        print(f"  Error: Could not open video writer for {temp_path}")
        if cap:
            cap.release()
        return

    for idx in tqdm(range(total), desc=f"  Rendering", leave=False):
        # Read video frame
        frame = None
        if cap:
            ret, video_frame = cap.read()
            if ret:
                vh, vw = video_frame.shape[:2]
                if vw != width or vh != height:
                    video_frame = cv2.resize(video_frame, (width, height))
                frame = video_frame

        if frame is None:
            import numpy as np
            frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Draw card detections
        detections = merged_card_data[idx].get('detections', [])
        frame = draw_card_detections(frame, detections)

        # Overlay action label
        label = get_label_at_frame(timeline_segments, idx)
        if label:
            cv2.putText(frame, f"Action: {label}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)

        # Frame counter + detection count
        n_dets = len(detections)
        cv2.putText(frame, f"Frame {idx}/{total}  Cards: {n_dets}", (30, height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)

        writer.write(frame)

    writer.release()
    if cap:
        cap.release()

    # Convert to h264
    cmd = [
        '/usr/bin/ffmpeg', '-y', '-i', temp_path,
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
        '-pix_fmt', 'yuv420p', final_path
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        os.remove(temp_path)
        print(f"  Video saved: {final_path}")
    except subprocess.CalledProcessError:
        print(f"  Warning: ffmpeg conversion failed, keeping temp file")
        os.rename(temp_path, final_path)


# ─── args ────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge per-300-frame card detection JSON files into per-round JSONL files"
    )
    parser.add_argument(
        '--examples-only', type=int, default=0, metavar='N',
        help='Process only N clips, merge + visualize them, then stop.'
    )
    parser.add_argument(
        '--vis-examples', type=int, default=0, metavar='N',
        help='After full processing, visualize N random successfully merged clips.'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for example selection (default: 42)'
    )
    parser.add_argument(
        '--width', type=int, default=1920,
        help='Visualization canvas width (default: 1920)'
    )
    parser.add_argument(
        '--height', type=int, default=1080,
        help='Visualization canvas height (default: 1080)'
    )
    return parser.parse_args()


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    random.seed(args.seed)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)

    # Step 1: Group card detection chunks by base video
    print("=" * 80)
    print("STEP 1: Grouping card detection chunk files")
    print("=" * 80)
    chunk_grouped = group_chunk_files(CARD_CHUNK_DIR)

    # Step 2: Get round names from prediction JSONs
    print("\n" + "=" * 80)
    print("STEP 2: Collecting round names from predictions")
    print("=" * 80)
    pred_files = sorted(glob.glob(os.path.join(PREDICTIONS_DIR, "*.json")))
    print(f"Found {len(pred_files)} prediction files")

    # --examples-only mode
    if args.examples_only > 0:
        # Filter to clips with available source videos
        eligible = []
        for pf in pred_files:
            try:
                bn, _, _ = parse_round_name(pf)
                if find_source_video(bn):
                    eligible.append(pf)
            except Exception:
                pass
        print(f"  {len(eligible)}/{len(pred_files)} clips have source videos")
        selected = random.sample(eligible, min(args.examples_only, len(eligible)))
        print(f"\n*** EXAMPLES-ONLY MODE: {len(selected)} clips ***\n")

        for pred_file in selected:
            round_stem = os.path.basename(pred_file).replace('.json', '')
            print(f"\n--- {round_stem} ---")

            try:
                base_name, round_start_1idx, round_end_1idx = parse_round_name(pred_file)
            except Exception:
                print(f"  Skipped: could not parse round name")
                continue

            global_start = round_start_1idx - 1
            global_end = round_end_1idx - 1

            chunk_group = chunk_grouped.get(base_name, [])
            if not chunk_group:
                print(f"  Skipped: no matching card chunks")
                continue

            merged = merge_chunks_for_round(chunk_group, global_start, global_end)
            if merged is None:
                print(f"  Skipped: no overlapping card chunks")
                continue

            # Save JSONL
            out_path = os.path.join(OUTPUT_DIR, f"{round_stem}_card.jsonl")
            with open(out_path, 'w') as f:
                for frame_data in merged:
                    f.write(json.dumps(frame_data) + '\n')
            print(f"  Merged: {len(merged)} frames -> {os.path.basename(out_path)}")

            # Visualize
            visualize_merged_round(
                merged, pred_file, VIS_DIR,
                width=args.width, height=args.height,
            )

        print(f"\nExamples done. Check visualizations in: {VIS_DIR}")
        return

    # Check already done (resume support)
    existing = {os.path.basename(f).replace('_card.jsonl', '')
                for f in glob.glob(os.path.join(OUTPUT_DIR, "*_card.jsonl"))}
    if existing:
        print(f"Resuming: {len(existing)} already done, skipping them")

    # Step 3: Merge
    print("\n" + "=" * 80)
    print("STEP 3: Merging card detections per round")
    print("=" * 80)
    n_success = 0
    n_no_match = 0
    n_skipped = 0
    success_paths = []

    for pred_file in tqdm(pred_files, desc="Merging"):
        round_stem = os.path.basename(pred_file).replace('.json', '')

        if round_stem in existing:
            n_skipped += 1
            # Still track for vis-examples
            out_path = os.path.join(OUTPUT_DIR, f"{round_stem}_card.jsonl")
            if os.path.exists(out_path):
                success_paths.append((out_path, pred_file))
            continue

        try:
            base_name, round_start_1idx, round_end_1idx = parse_round_name(pred_file)
        except Exception:
            n_no_match += 1
            continue

        global_start = round_start_1idx - 1  # 0-indexed
        global_end = round_end_1idx - 1

        chunk_group = chunk_grouped.get(base_name, [])
        if not chunk_group:
            n_no_match += 1
            continue

        merged = merge_chunks_for_round(chunk_group, global_start, global_end)
        if merged is None:
            n_no_match += 1
            continue

        # Write as JSONL (one line per frame)
        out_path = os.path.join(OUTPUT_DIR, f"{round_stem}_card.jsonl")
        with open(out_path, 'w') as f:
            for frame_data in merged:
                f.write(json.dumps(frame_data) + '\n')

        n_success += 1
        success_paths.append((out_path, pred_file))

    print(f"\n{'=' * 80}")
    print(f"SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total rounds: {len(pred_files)}")
    print(f"Successfully merged: {n_success}")
    print(f"No matching chunks: {n_no_match}")
    print(f"Skipped (resume): {n_skipped}")
    print(f"Output: {OUTPUT_DIR}")

    # Step 4: Visualize examples (if requested)
    if args.vis_examples > 0 and success_paths:
        n_vis = min(args.vis_examples, len(success_paths))
        vis_selection = random.sample(success_paths, n_vis)

        print(f"\n{'='*80}")
        print(f"STEP 4: Visualizing {n_vis} examples")
        print(f"{'='*80}")

        for jsonl_path, pred_file in vis_selection:
            # Load merged card data from JSONL
            merged_data = []
            with open(jsonl_path) as f:
                for line in f:
                    merged_data.append(json.loads(line))

            visualize_merged_round(
                merged_data, pred_file, VIS_DIR,
                width=args.width, height=args.height,
            )

        print(f"\nVisualizations saved to: {VIS_DIR}")


if __name__ == "__main__":
    main()
