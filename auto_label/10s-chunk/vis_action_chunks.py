#!/usr/bin/env python3
"""
Visualize individual action chunks (segments) from source video.
Generates ~500 clips per action category, saved to per-category subdirectories.

Usage:
    python vis_action_chunks.py [--per-category 500] [--seed 42]
"""

import os
import sys
import json
import glob
import subprocess
import argparse
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PREDICTION_DIR = "/home/ubuntu/us-west-3-fs/sahithi/hard_action_annotations/predictions_json"
VIDEO_DIR = "/home/ubuntu/us-west-3-fs/live_dealer_blackjack/raw/batch_01"
OUTPUT_BASE = "/home/ubuntu/us-west-3-fs/sahithi/vis"
WIDTH = 1920
HEIGHT = 1080
FPS = 30


def parse_clip_name(name: str) -> Tuple[str, int, int]:
    for suffix in ['_annotations.json', '_predictions.json', '.json']:
        name = name.replace(suffix, '')
    basename = os.path.basename(name)
    parts = basename.split('_')
    if len(parts) != 4:
        raise ValueError(f"Invalid clip name: {name}")
    base_name = f"{parts[0]}_{parts[1]}"
    clip_start = int(parts[2])
    clip_end = int(parts[3])
    return base_name, clip_start, clip_end


def find_source_video(base_name: str) -> Optional[str]:
    video_filename = base_name.replace('_', ' ', 1) + ".mp4"
    video_path = os.path.join(VIDEO_DIR, video_filename)
    if os.path.exists(video_path):
        return video_path
    return None


def collect_action_segments(prediction_dir: str) -> Dict[str, List[dict]]:
    """Collect all action segments grouped by label.
    Each entry: {json_path, base_name, clip_start_1idx, clip_end_1idx, seg_start, seg_end, label}
    """
    segments_by_label = defaultdict(list)
    json_files = sorted(glob.glob(os.path.join(prediction_dir, "*.json")))
    print(f"Scanning {len(json_files)} prediction files for action segments...")

    for json_path in tqdm(json_files, desc="Collecting segments"):
        try:
            base_name, clip_start, clip_end = parse_clip_name(json_path)
        except Exception:
            continue

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except Exception:
            continue

        for seg in data.get('timeline_segments', []):
            labels = seg.get('labels', [])
            if not labels:
                continue
            label = labels[0]
            segments_by_label[label].append({
                'json_path': json_path,
                'base_name': base_name,
                'clip_start_1idx': clip_start,
                'clip_end_1idx': clip_end,
                'seg_start': seg['start_frame'],
                'seg_end': seg['end_frame'],
                'label': label,
            })

    for label, segs in segments_by_label.items():
        print(f"  {label:<30} {len(segs):>6} segments")
    return dict(segments_by_label)


def visualize_action_chunk(seg_info: dict, out_dir: str) -> bool:
    """Extract action segment frames from source video, with label overlay.
    Returns True on success."""
    import cv2

    base_name = seg_info['base_name']
    seg_start = seg_info['seg_start']  # relative to clip
    seg_end = seg_info['seg_end']      # relative to clip
    clip_start_1idx = seg_info['clip_start_1idx']
    label = seg_info['label']

    # Output filename
    clip_stem = os.path.basename(seg_info['json_path']).replace('.json', '')
    out_filename = f"{clip_stem}_f{seg_start:06d}_{seg_end:06d}.mp4"
    final_path = os.path.join(out_dir, out_filename)
    if os.path.exists(final_path):
        return True

    # Open source video
    video_path = find_source_video(base_name)
    if not video_path:
        return False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False

    global_start_frame = (clip_start_1idx - 1) + seg_start  # 0-indexed
    cap.set(cv2.CAP_PROP_POS_FRAMES, global_start_frame)

    n_frames = seg_end - seg_start + 1
    if n_frames <= 0:
        cap.release()
        return False

    # Write video
    temp_path = final_path.replace('.mp4', '_temp.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(temp_path, fourcc, FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        cap.release()
        return False

    for idx in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        vh, vw = frame.shape[:2]
        if vw != WIDTH or vh != HEIGHT:
            frame = cv2.resize(frame, (WIDTH, HEIGHT))

        # Label overlay
        cv2.putText(frame, f"{label}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, f"Frame {idx}/{n_frames}", (30, HEIGHT - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)

        writer.write(frame)

    writer.release()
    cap.release()

    # Convert to h264
    cmd = [
        '/usr/bin/ffmpeg', '-y', '-i', temp_path,
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-pix_fmt', 'yuv420p', final_path
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        os.remove(temp_path)
    except subprocess.CalledProcessError:
        if os.path.exists(temp_path):
            os.rename(temp_path, final_path)

    return True


def main():
    parser = argparse.ArgumentParser(description="Visualize action chunks with dwpose on video")
    parser.add_argument('--per-category', type=int, default=500, help='Max clips per category')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)

    # Collect all segments
    segments_by_label = collect_action_segments(PREDICTION_DIR)

    # Pre-compute which base_names have source videos
    print("\nFiltering segments to those with available source videos...")
    available_videos = set()
    all_base_names = set()
    for segs in segments_by_label.values():
        for s in segs:
            all_base_names.add(s['base_name'])
    for bn in all_base_names:
        if find_source_video(bn):
            available_videos.add(bn)
    print(f"  {len(available_videos)}/{len(all_base_names)} base videos have raw video files")

    # Sample and visualize per category
    for label, all_segs in sorted(segments_by_label.items()):
        if label == 'background':
            print(f"\nSkipping 'background' category")
            continue

        # Filter to segments with available source video
        eligible = [s for s in all_segs if s['base_name'] in available_videos]
        print(f"\n  {label}: {len(eligible)}/{len(all_segs)} segments have source video")

        safe_label = label.replace(' ', '_')
        out_dir = os.path.join(OUTPUT_BASE, safe_label)
        os.makedirs(out_dir, exist_ok=True)

        n_sample = min(args.per_category, len(eligible))
        sampled = random.sample(eligible, n_sample)
        print(f"\n{'='*60}")
        print(f"Category: {label} — visualizing {n_sample}/{len(all_segs)} segments")
        print(f"Output: {out_dir}")
        print(f"{'='*60}")

        success = 0
        for seg_info in tqdm(sampled, desc=f"  {label}"):
            if visualize_action_chunk(seg_info, out_dir):
                success += 1
        print(f"  Done: {success}/{n_sample} clips saved")


if __name__ == "__main__":
    main()
