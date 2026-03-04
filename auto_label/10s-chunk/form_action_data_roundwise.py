#!/usr/bin/env python3
"""
Merge per-300-frame dwpose pkl files into per-round pkl files.

For each annotation/prediction JSON (which defines a video clip YYYY-MM-DD_HH-MM-SS_XXXXXX_YYYYYY):
1. Find all dwpose pkl files that overlap with the clip's global frame range
2. Concatenate them in order, trimming head and tail to the clip boundaries
3. Save the merged result as a single pkl in the SAME format as the originals
4. Optionally visualize with draw_pose and overlay action labels from the JSON

Usage:
    # Examples-only mode (process + visualize a few clips first):
    python form_action_data_roundwise.py --examples-only 3

    # Full processing (no visualization):
    python form_action_data_roundwise.py

    # Full processing + visualize N random examples:
    python form_action_data_roundwise.py --vis-examples 5

Requires: conda activate mdmpy310
"""

import os
import sys
import json
import pickle
import glob
import subprocess
import argparse
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from tqdm import tqdm

# Compatibility shim: PKL files saved with numpy >=2.0 use numpy._core,
# but this environment has numpy 1.x which only has numpy.core.
if not hasattr(np, '_core'):
    import numpy.core as _np_core
    import sys as _sys
    _sys.modules.setdefault('numpy._core', _np_core)
    for _name in dir(_np_core):
        _sys.modules.setdefault(f'numpy._core.{_name}', getattr(_np_core, _name))

# For draw_pose visualization
sys.path.insert(0, str(Path("/home/ubuntu/yifan/code/cleanpull/motion-diffusion-model/data_loaders/humanml/utils")))
from plot_script import draw_pose


FRAMES_PER_PKL = 300


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def parse_pkl_filename(filename: str) -> Tuple[str, int, int]:
    """
    Parse dwpose pkl filename.

    Format: YYYY-MM-DD_HH-MM-SS_XXXXXX_YYYYYY_dwpose.pkl
    Returns: (base_name, clip_index, start_frame_0indexed)
    """
    basename = os.path.basename(filename).replace('_dwpose.pkl', '')
    parts = basename.split('_')
    if len(parts) != 4:
        raise ValueError(f"Invalid pkl filename: {filename}")
    base_name = f"{parts[0]}_{parts[1]}"
    clip_index = int(parts[2])
    start_frame = int(parts[3])
    return base_name, clip_index, start_frame


def parse_clip_name(name: str) -> Tuple[str, int, int]:
    """
    Parse clip identifier from JSON filename.

    Handles suffixes: _annotations.json, _predictions.json
    XXXXXX / YYYYYY are 1-indexed.
    Returns: (base_name, clip_start_1indexed, clip_end_1indexed)
    """
    for suffix in ['_annotations.json', '_predictions.json']:
        name = name.replace(suffix, '')
    basename = os.path.basename(name)
    parts = basename.split('_')
    if len(parts) != 4:
        raise ValueError(f"Invalid clip name: {name}")
    base_name = f"{parts[0]}_{parts[1]}"
    clip_start = int(parts[2])  # 1-indexed
    clip_end = int(parts[3])    # 1-indexed
    return base_name, clip_start, clip_end


# ---------------------------------------------------------------------------
# Group pkl files by base video
# ---------------------------------------------------------------------------

def group_pkl_files(pkl_dir: str) -> Dict[str, List[Tuple[str, int, int]]]:
    """
    Returns: base_name -> sorted list of (filepath, clip_index, start_frame_0indexed)
    """
    grouped = defaultdict(list)
    pkl_pattern = os.path.join(pkl_dir, "*_dwpose.pkl")
    pkl_files = glob.glob(pkl_pattern)
    print(f"Found {len(pkl_files)} pkl files in {pkl_dir}")

    for filepath in tqdm(pkl_files, desc="Parsing pkl filenames"):
        try:
            base_name, clip_index, start_frame = parse_pkl_filename(filepath)
            grouped[base_name].append((filepath, clip_index, start_frame))
        except Exception as e:
            pass  # silently skip unparseable files during grouping

    for base_name in grouped:
        grouped[base_name].sort(key=lambda x: x[2])

    print(f"Grouped into {len(grouped)} base videos")
    return grouped


# ---------------------------------------------------------------------------
# Merge logic
# ---------------------------------------------------------------------------

def merge_pkls_for_clip(
    pkl_group: List[Tuple[str, int, int]],
    global_start_0idx: int,
    global_end_0idx: int,
) -> Tuple[Optional[list], bool]:
    """
    Merge dwpose pkl data for a clip's global frame range.

    Returns:
        (merged_frames, is_full_coverage)
        merged_frames is None if no overlapping files found.
    """
    total_frames_needed = global_end_0idx - global_start_0idx + 1

    matching = []
    for filepath, clip_index, pkl_start in pkl_group:
        pkl_end = pkl_start + FRAMES_PER_PKL - 1
        if pkl_end >= global_start_0idx and pkl_start <= global_end_0idx:
            matching.append((filepath, pkl_start, pkl_end))

    if not matching:
        return None, False

    matching.sort(key=lambda x: x[1])

    # Coverage check
    is_full = (matching[0][1] <= global_start_0idx and
               matching[-1][2] >= global_end_0idx)
    if is_full:
        for i in range(len(matching) - 1):
            if matching[i + 1][1] > matching[i][2] + 1:
                is_full = False
                break

    merged_frames = []
    for filepath, pkl_start, pkl_end in matching:
        with open(filepath, 'rb') as f:
            frames = pickle.load(f)
        slice_start = max(0, global_start_0idx - pkl_start)
        slice_end = min(len(frames), global_end_0idx - pkl_start + 1)
        merged_frames.extend(frames[slice_start:slice_end])

    return merged_frames, is_full


# ---------------------------------------------------------------------------
# Process one JSON file
# ---------------------------------------------------------------------------

def count_labels_in_json(json_path: str) -> Dict[str, int]:
    """Read a JSON file and count action labels in timeline_segments."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        counts = defaultdict(int)
        for seg in data.get('timeline_segments', []):
            for label in seg.get('labels', []):
                counts[label] += 1
        return dict(counts)
    except Exception:
        return {}


def process_json_file(
    json_path: str,
    pkl_grouped: Dict[str, List[Tuple[str, int, int]]],
    output_dir: str,
) -> Dict[str, object]:
    """
    Merge dwpose pkls for one annotation/prediction clip.

    Returns: {'success': 0|1, 'no_match': 0|1, 'partial': 0|1,
              'n_frames': int, 'expected_frames': int,
              'clip_stem': str, 'base_name': str, 'out_path': str|None}
    """
    basename = os.path.basename(json_path)
    result = {
        'success': 0, 'no_match': 0, 'partial': 0,
        'n_frames': 0, 'expected_frames': 0,
        'clip_stem': '', 'base_name': '', 'out_path': None,
    }

    try:
        base_name, clip_start_1idx, clip_end_1idx = parse_clip_name(json_path)
    except Exception as e:
        result['no_match'] = 1
        return result

    global_start = clip_start_1idx - 1
    global_end = clip_end_1idx - 1
    total_frames = global_end - global_start + 1

    clip_stem = basename.replace('_annotations.json', '').replace('_predictions.json', '')
    result['clip_stem'] = clip_stem
    result['base_name'] = base_name
    result['expected_frames'] = total_frames

    out_filename = f"{clip_stem}_dwpose.pkl"
    out_path = os.path.join(output_dir, out_filename)

    # Resume support: skip files already written in a previous run.
    if os.path.exists(out_path):
        result['success'] = 1
        result['out_path'] = out_path
        return result

    pkl_group = pkl_grouped.get(base_name, [])
    if not pkl_group:
        result['no_match'] = 1
        return result

    merged, is_full = merge_pkls_for_clip(pkl_group, global_start, global_end)
    if merged is None:
        result['no_match'] = 1
        return result

    is_partial = not is_full or len(merged) < total_frames
    result['n_frames'] = len(merged)

    with open(out_path, 'wb') as f:
        pickle.dump(merged, f)

    result['success'] = 1
    result['partial'] = 1 if is_partial else 0
    result['out_path'] = out_path
    return result


# ---------------------------------------------------------------------------
# Visualization using draw_pose + annotation overlay
# ---------------------------------------------------------------------------

def get_label_at_frame(timeline_segments: list, frame_idx: int) -> str:
    """Return the action label for the given relative frame index, or '' if none."""
    for seg in timeline_segments:
        if seg['start_frame'] <= frame_idx <= seg['end_frame']:
            labels = seg.get('labels', [])
            return labels[0] if labels else ''
    return ''


# Color map for labels
LABEL_COLORS = {
    'close bets':        (0, 200, 255),   # orange
    'initial hands 1st': (0, 255, 0),     # green
    'initial hands 2nd': (0, 255, 128),   # light green
    'reveal hole card':  (255, 100, 100), # blue-ish
    'discard':           (100, 100, 255), # red-ish
    'background':        (128, 128, 128), # gray
}


def get_label_color(label: str) -> Tuple[int, int, int]:
    return LABEL_COLORS.get(label, (200, 200, 200))


def visualize_merged_round(
    pkl_path: str,
    json_path: str,
    vis_dir: str,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
):
    """
    Visualize a merged round pkl as an mp4 video with draw_pose and annotation overlay.

    Args:
        pkl_path: path to merged dwpose pkl
        json_path: path to the annotation/prediction JSON (for label overlay)
        vis_dir: output directory
        width: canvas width for draw_pose
        height: canvas height for draw_pose
        fps: output video fps
    """
    import cv2

    with open(pkl_path, 'rb') as f:
        frames = pickle.load(f)

    # Load annotation for label overlay
    with open(json_path, 'r') as f:
        annotation = json.load(f)
    timeline_segments = annotation.get('timeline_segments', [])

    basename = os.path.basename(pkl_path).replace('.pkl', '')
    total = len(frames)
    print(f"  Visualizing {basename}: {total} frames")

    # Video output
    temp_path = os.path.join(vis_dir, f"{basename}_vis_temp.mp4")
    final_path = os.path.join(vis_dir, f"{basename}_vis.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        print(f"  Error: Could not open video writer for {temp_path}")
        return

    for idx in tqdm(range(total), desc=f"  Rendering", leave=False):
        item = frames[idx]
        pose = item['pose']

        # Use draw_pose for proper skeleton rendering (returns RGB)
        canvas = draw_pose(pose, H=height, W=width)

        # Convert RGB -> BGR for cv2
        canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

        # Overlay action label from annotation
        label = get_label_at_frame(timeline_segments, idx)
        label_color = get_label_color(label)
        if label:
            cv2.putText(canvas, f"Action: {label}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, label_color, 3, cv2.LINE_AA)

        # Frame counter
        cv2.putText(canvas, f"Frame {idx}/{total}", (30, height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)

        writer.write(canvas)

    writer.release()

    # Convert to h264
    cmd = [
        'ffmpeg', '-y', '-i', temp_path,
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


# ---------------------------------------------------------------------------
# Stats export
# ---------------------------------------------------------------------------

def write_stats(
    stats_path: str,
    annot_results: list,
    pred_results: list,
    annot_label_counts: Dict[str, int],
    pred_label_counts: Dict[str, int],
):
    """Write processing statistics to a text file."""

    def _section_stats(results: list, label_counts: Dict[str, int], source_name: str, f):
        total_success = sum(r['success'] for r in results)
        total_no_match = sum(r['no_match'] for r in results)
        total_partial = sum(r['partial'] for r in results)

        f.write(f"\n{'='*80}\n")
        f.write(f"{source_name}\n")
        f.write(f"{'='*80}\n")
        f.write(f"Clips processed: {len(results)}\n")
        f.write(f"Merged PKL files created: {total_success}\n")
        f.write(f"  - With full coverage: {total_success - total_partial}\n")
        f.write(f"  - With partial coverage: {total_partial}\n")
        f.write(f"Skipped - no matching pkl files: {total_no_match}\n\n")

        # Action label distribution
        f.write(f"Action label distribution ({source_name}):\n")
        f.write(f"{'-'*50}\n")
        total_actions = sum(label_counts.values())
        for label in sorted(label_counts.keys()):
            count = label_counts[label]
            pct = count / total_actions * 100 if total_actions > 0 else 0
            f.write(f"  {label:<30} {count:>6}  ({pct:5.1f}%)\n")
        f.write(f"  {'TOTAL':<30} {total_actions:>6}\n\n")

        # Per-file details
        f.write(f"Per-file details:\n")
        f.write(f"{'Clip':<55} {'Status':<10} {'Frames':<15} {'Expected':<10}\n")
        f.write(f"{'-'*90}\n")
        for r in sorted(results, key=lambda x: x['clip_stem']):
            if r['success']:
                status = 'partial' if r['partial'] else 'full'
            elif r['no_match']:
                status = 'no_match'
            else:
                status = 'error'
            f.write(f"{r['clip_stem']:<55} {status:<10} {r['n_frames']:<15} {r['expected_frames']:<10}\n")

    # Merge label counts for overall summary
    all_label_counts = defaultdict(int)
    for label, count in annot_label_counts.items():
        all_label_counts[label] += count
    for label, count in pred_label_counts.items():
        all_label_counts[label] += count

    with open(stats_path, 'w') as f:
        f.write(f"form_action_data_roundwise.py - Run Statistics\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n\n")

        all_results = annot_results + pred_results
        total_success = sum(r['success'] for r in all_results)
        total_no_match = sum(r['no_match'] for r in all_results)
        total_partial = sum(r['partial'] for r in all_results)

        f.write(f"Overall Summary:\n")
        f.write(f"  Total clips processed: {len(all_results)}\n")
        f.write(f"    - From annotation files: {len(annot_results)}\n")
        f.write(f"    - From prediction files: {len(pred_results)}\n")
        f.write(f"  Merged PKL files created: {total_success}\n")
        f.write(f"    - With full coverage: {total_success - total_partial}\n")
        f.write(f"    - With partial coverage: {total_partial}\n")
        f.write(f"  Skipped - no matching pkl files: {total_no_match}\n\n")

        # Overall action label distribution
        f.write(f"Overall action label distribution:\n")
        f.write(f"{'-'*50}\n")
        total_actions = sum(all_label_counts.values())
        for label in sorted(all_label_counts.keys()):
            count = all_label_counts[label]
            pct = count / total_actions * 100 if total_actions > 0 else 0
            f.write(f"  {label:<30} {count:>6}  ({pct:5.1f}%)\n")
        f.write(f"  {'TOTAL':<30} {total_actions:>6}\n")

        # Per-source sections
        _section_stats(annot_results, annot_label_counts, "ANNOTATIONS", f)
        _section_stats(pred_results, pred_label_counts, "PREDICTIONS", f)

    print(f"Stats written to: {stats_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge per-300-frame dwpose pkl files into per-round pkl files"
    )
    parser.add_argument(
        '--examples-only', type=int, default=0, metavar='N',
        help='Process only N clips, merge + visualize them, then stop. '
             'Use this to check results before full run.'
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


def main():
    args = parse_args()
    random.seed(args.seed)


    PKL_DIR = "/home/ubuntu/us-west-3-fs/live_dealer_blackjack/dwpose/batch_01/dwpose"
    ANNOTATION_DIR = ""
    PREDICTION_DIR = "/home/ubuntu/yifan/code/motion-data-process/auto_label/10s-chunk/data/inference_results_filtered_videos_26k/predictions_json_with_meta"
    OUTPUT_BASE = "/home/ubuntu/us-west-3-fs/live_dealer_blackjack/action_annotation/mixed_mini_batch/dwpose/AutoLabeling_batch_01_part_1"
    OUTPUT_ANNOT = os.path.join(OUTPUT_BASE, "annotations")
    OUTPUT_PRED = os.path.join(OUTPUT_BASE, "predictions")
    VIS_DIR = "/home/ubuntu/yifan/code/motion-data-process/auto_label/10s-chunk/data/inference_results_filtered_videos_26k"
    STATS_PATH = "/home/ubuntu/yifan/code/motion-data-process/auto_label/10s-chunk/data/inference_results_filtered_videos_26k/merged_dwpose_stats.txt"

    os.makedirs(OUTPUT_ANNOT, exist_ok=True)
    os.makedirs(OUTPUT_PRED, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)

    # Step 1: Group pkl files
    print("=" * 80)
    print("STEP 1: Grouping PKL files")
    print("=" * 80)
    pkl_grouped = group_pkl_files(PKL_DIR)

    # Step 2: Collect all JSON files
    print("\n" + "=" * 80)
    print("STEP 2: Collecting annotation and prediction files")
    print("=" * 80)

    annotation_files = (
        sorted(glob.glob(os.path.join(ANNOTATION_DIR, "*_annotations.json")))
        if ANNOTATION_DIR else []
    )
    prediction_files = (
        sorted(glob.glob(os.path.join(PREDICTION_DIR, "*_predictions.json")))
        if PREDICTION_DIR else []
    )
    print(f"Found {len(annotation_files)} annotation files"
          + (f" in {ANNOTATION_DIR}" if ANNOTATION_DIR else " (ANNOTATION_DIR not set)"))
    print(f"Found {len(prediction_files)} prediction files"
          + (f" in {PREDICTION_DIR}" if PREDICTION_DIR else " (PREDICTION_DIR not set)"))

    # --examples-only mode: pick a few from each, process + visualize, then exit
    if args.examples_only > 0:
        n_each = max(1, args.examples_only // 2)
        n_annot_ex = min(n_each, len(annotation_files))
        n_pred_ex = min(args.examples_only - n_annot_ex, len(prediction_files))
        selected_annot = random.sample(annotation_files, n_annot_ex) if n_annot_ex > 0 else []
        selected_pred = random.sample(prediction_files, n_pred_ex) if n_pred_ex > 0 else []
        print(f"\n*** EXAMPLES-ONLY MODE: {n_annot_ex} annotations + {n_pred_ex} predictions ***\n")

        for json_path, out_dir, source in (
            [(p, OUTPUT_ANNOT, 'annotation') for p in selected_annot] +
            [(p, OUTPUT_PRED, 'prediction') for p in selected_pred]
        ):
            basename = os.path.basename(json_path)
            print(f"\n--- [{source}] {basename} ---")
            result = process_json_file(json_path, pkl_grouped, out_dir)

            if result['success']:
                status = 'PARTIAL' if result['partial'] else 'FULL'
                print(f"  Status: {status} ({result['n_frames']}/{result['expected_frames']} frames)")
                print(f"  Saved: {os.path.basename(result['out_path'])}")

                # Visualize
                visualize_merged_round(
                    result['out_path'], json_path, VIS_DIR,
                    width=args.width, height=args.height,
                )
            else:
                print(f"  Status: NO MATCH")

        print(f"\nExamples done. Check visualizations in: {VIS_DIR}")
        return

    # Step 3: Full processing - annotations and predictions separately
    def process_source(json_files, output_dir, source_name):
        """Process a list of JSON files, return (results, label_counts, successful_paths)."""
        print(f"\n--- Processing {source_name} ({len(json_files)} files) ---")
        results = []
        label_counts = defaultdict(int)
        success_paths = []

        for json_path in tqdm(json_files, desc=f"Merging {source_name}"):
            # Count labels from JSON
            lc = count_labels_in_json(json_path)
            for label, count in lc.items():
                label_counts[label] += count

            result = process_json_file(json_path, pkl_grouped, output_dir)
            results.append(result)
            if result['success'] and result['out_path']:
                success_paths.append((result['out_path'], json_path))

        return results, dict(label_counts), success_paths

    print("\n" + "=" * 80)
    print("STEP 3: Processing all clips")
    print("=" * 80)

    annot_results, annot_label_counts, annot_success_paths = process_source(
        annotation_files, OUTPUT_ANNOT, "annotations"
    )
    pred_results, pred_label_counts, pred_success_paths = process_source(
        prediction_files, OUTPUT_PRED, "predictions"
    )

    all_results = annot_results + pred_results
    successful_paths = annot_success_paths + pred_success_paths

    # Summary
    total_success = sum(r['success'] for r in all_results)
    total_no_match = sum(r['no_match'] for r in all_results)
    total_partial = sum(r['partial'] for r in all_results)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total clips processed: {len(all_results)}")
    print(f"  - Annotation files: {len(annot_results)}")
    print(f"  - Prediction files: {len(pred_results)}")
    print(f"Merged PKL files created: {total_success}")
    print(f"  - With full coverage: {total_success - total_partial}")
    print(f"  - With partial coverage: {total_partial}")
    print(f"Skipped - no matching pkl files: {total_no_match}")
    print(f"\nAction label counts (annotations):")
    for label in sorted(annot_label_counts.keys()):
        print(f"  {label:<30} {annot_label_counts[label]:>6}")
    print(f"Action label counts (predictions):")
    for label in sorted(pred_label_counts.keys()):
        print(f"  {label:<30} {pred_label_counts[label]:>6}")
    print(f"\nOutput directories:")
    print(f"  Annotations: {OUTPUT_ANNOT}")
    print(f"  Predictions: {OUTPUT_PRED}")

    # Write stats file
    write_stats(STATS_PATH, annot_results, pred_results,
                annot_label_counts, pred_label_counts)

    # Step 4: Visualize examples (if requested)
    if args.vis_examples > 0 and successful_paths:
        n_vis = min(args.vis_examples, len(successful_paths))
        vis_selection = random.sample(successful_paths, n_vis)

        print(f"\n{'='*80}")
        print(f"STEP 4: Visualizing {n_vis} examples")
        print(f"{'='*80}")

        for pkl_path, json_path in vis_selection:
            visualize_merged_round(
                pkl_path, json_path, VIS_DIR,
                width=args.width, height=args.height,
            )

        print(f"\nVisualizations saved to: {VIS_DIR}")


if __name__ == "__main__":
    main()
