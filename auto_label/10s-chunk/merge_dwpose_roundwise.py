#!/usr/bin/env python3
"""
Merge per-300-frame dwpose pkl files into per-round pkl files.

Simplified version of form_action_data_roundwise.py (no visualization).

Usage:
    python merge_dwpose_roundwise.py
"""

import os
import glob
import pickle
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
from tqdm import tqdm

# Compatibility shim for numpy 2.0+ pickles
if not hasattr(np, '_core'):
    import numpy.core as _np_core
    import sys as _sys
    _sys.modules.setdefault('numpy._core', _np_core)
    for _name in dir(_np_core):
        _sys.modules.setdefault(f'numpy._core.{_name}', getattr(_np_core, _name))

# ─── paths ────────────────────────────────────────────────────────────────────
PKL_DIR = "/home/ubuntu/us-west-3-fs/live_dealer_blackjack/dwpose/batch_01/dwpose"
PREDICTIONS_DIR = (
    "/home/ubuntu/us-west-3-fs/sahithi/hard_action_annotations/predictions_json"
)
OUTPUT_DIR = (
    "/home/ubuntu/us-west-3-fs/sahithi/hard_action_annotations/dwpose_roundwise"
)

FRAMES_PER_PKL = 300


# ─── filename parsing ─────────────────────────────────────────────────────────

def parse_pkl_filename(filename: str) -> Tuple[str, int, int]:
    basename = os.path.basename(filename).replace('_dwpose.pkl', '')
    parts = basename.split('_')
    if len(parts) != 4:
        raise ValueError(f"Invalid pkl filename: {filename}")
    base_name = f"{parts[0]}_{parts[1]}"
    clip_index = int(parts[2])
    start_frame = int(parts[3])
    return base_name, clip_index, start_frame


def parse_round_name(name: str) -> Tuple[str, int, int]:
    basename = os.path.basename(name).replace('.json', '')
    parts = basename.split('_')
    if len(parts) != 4:
        raise ValueError(f"Invalid round name: {name}")
    base_name = f"{parts[0]}_{parts[1]}"
    round_start = int(parts[2])  # 1-indexed
    round_end = int(parts[3])    # 1-indexed
    return base_name, round_start, round_end


# ─── group pkl files ─────────────────────────────────────────────────────────

def group_pkl_files(pkl_dir: str) -> Dict[str, List[Tuple[str, int, int]]]:
    grouped = defaultdict(list)
    pkl_files = glob.glob(os.path.join(pkl_dir, "*_dwpose.pkl"))
    print(f"Found {len(pkl_files)} pkl files in {pkl_dir}")

    for filepath in tqdm(pkl_files, desc="Parsing pkl filenames"):
        try:
            base_name, clip_index, start_frame = parse_pkl_filename(filepath)
            grouped[base_name].append((filepath, clip_index, start_frame))
        except Exception:
            pass

    for base_name in grouped:
        grouped[base_name].sort(key=lambda x: x[2])

    print(f"Grouped into {len(grouped)} base videos")
    return grouped


# ─── merge logic ─────────────────────────────────────────────────────────────

def merge_pkls_for_round(
    pkl_group: List[Tuple[str, int, int]],
    global_start_0idx: int,
    global_end_0idx: int,
) -> Optional[list]:
    matching = []
    for filepath, clip_index, pkl_start in pkl_group:
        pkl_end = pkl_start + FRAMES_PER_PKL - 1
        if pkl_end >= global_start_0idx and pkl_start <= global_end_0idx:
            matching.append((filepath, pkl_start, pkl_end))

    if not matching:
        return None

    matching.sort(key=lambda x: x[1])

    merged_frames = []
    for filepath, pkl_start, pkl_end in matching:
        with open(filepath, 'rb') as f:
            frames = pickle.load(f)
        slice_start = max(0, global_start_0idx - pkl_start)
        slice_end = min(len(frames), global_end_0idx - pkl_start + 1)
        merged_frames.extend(frames[slice_start:slice_end])

    return merged_frames


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Group pkl files
    print("=" * 80)
    print("STEP 1: Grouping PKL files")
    print("=" * 80)
    pkl_grouped = group_pkl_files(PKL_DIR)

    # Step 2: Get round names from predictions
    print("\n" + "=" * 80)
    print("STEP 2: Collecting round names from predictions")
    print("=" * 80)
    pred_files = sorted(glob.glob(os.path.join(PREDICTIONS_DIR, "*.json")))
    print(f"Found {len(pred_files)} prediction files")

    # Resume support
    existing = {os.path.basename(f).replace('_dwpose.pkl', '')
                for f in glob.glob(os.path.join(OUTPUT_DIR, "*_dwpose.pkl"))}
    if existing:
        print(f"Resuming: {len(existing)} already done, skipping them")

    # Step 3: Merge
    print("\n" + "=" * 80)
    print("STEP 3: Merging dwpose per round")
    print("=" * 80)
    n_success = 0
    n_no_match = 0
    n_skipped = 0

    for pred_file in tqdm(pred_files, desc="Merging"):
        round_stem = os.path.basename(pred_file).replace('.json', '')

        if round_stem in existing:
            n_skipped += 1
            continue

        try:
            base_name, round_start_1idx, round_end_1idx = parse_round_name(pred_file)
        except Exception:
            n_no_match += 1
            continue

        global_start = round_start_1idx - 1
        global_end = round_end_1idx - 1

        pkl_group = pkl_grouped.get(base_name, [])
        if not pkl_group:
            n_no_match += 1
            continue

        merged = merge_pkls_for_round(pkl_group, global_start, global_end)
        if merged is None:
            n_no_match += 1
            continue

        out_path = os.path.join(OUTPUT_DIR, f"{round_stem}_dwpose.pkl")
        with open(out_path, 'wb') as f:
            pickle.dump(merged, f)

        n_success += 1

    print(f"\n{'=' * 80}")
    print(f"SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total rounds: {len(pred_files)}")
    print(f"Successfully merged: {n_success}")
    print(f"No matching chunks: {n_no_match}")
    print(f"Skipped (resume): {n_skipped}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
