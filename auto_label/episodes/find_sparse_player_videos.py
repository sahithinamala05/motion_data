"""
Find roundcut videos with <=N active players using card detection clustering.

Scans card detections across all frames, clusters into player seats,
and saves a representative frame for videos with 1-N active players.

Usage:
    python find_sparse_player_videos.py [--max_players N] [--workers N]
"""

import argparse
import json
import multiprocessing as mp
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "/home/ubuntu/sahithi/motion-data-process/auto_label/10s-chunk")
from clustering_util import assign_cards_to_positions

# ─── Paths ───────────────────────────────────────────────────────────────────
CARDS_DIR = (
    "/home/ubuntu/us-west-3-fs/sahithi/hard_action_annotations/card_detections_roundwise"
)
RAW_VIDEO_DIR = "/home/ubuntu/us-west-3-fs/live_dealer_blackjack/raw/batch_01"
GOOD_ROUNDS_DIR = (
    "/home/ubuntu/us-west-3-fs/live_dealer_blackjack/roundcut/good_quality_rounds"
)
OUT_DIR = "/home/ubuntu/us-west-3-fs/sahithi/clus_anno/sparse_players"

IMG_W, IMG_H = 1280, 720


def _get_raw_video(date_time):
    base = date_time.replace("_", " ", 1)
    for name in (f"{base}.mp4", f"{date_time}.mp4"):
        path = os.path.join(RAW_VIDEO_DIR, name)
        if os.path.exists(path):
            return path
    return None


def _process_one(args):
    vname, max_players = args
    card_jsonl = os.path.join(CARDS_DIR, f"{vname}_card.jsonl")
    if not os.path.exists(card_jsonl):
        return None

    max_active = 0
    best_frame_id = None
    best_dets_count = 0
    with open(card_jsonl) as f:
        for line in f:
            obj = json.loads(line)
            if not obj["detections"]:
                continue
            dets = obj["detections"]
            for d in dets:
                if isinstance(d["polygon_center"][0], list):
                    d["polygon_center"] = d["polygon_center"][0]
            clusters = assign_cards_to_positions(dets)
            active = sum(1 for p in range(1, 8) if clusters[p])
            if active > max_active:
                max_active = active
            if len(dets) > best_dets_count:
                best_dets_count = len(dets)
                best_frame_id = obj["frame_id"]

    if max_active < 1 or max_active > max_players:
        return None

    parts = vname.split("_")
    date_time = parts[0] + "_" + parts[1]
    clip_start = int(parts[2])
    raw_path = _get_raw_video(date_time)
    if not raw_path:
        return None

    cap = cv2.VideoCapture(raw_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, (clip_start - 1) + best_frame_id)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    if frame.shape[1] != IMG_W or frame.shape[0] != IMG_H:
        frame = cv2.resize(frame, (IMG_W, IMG_H))

    out_path = os.path.join(OUT_DIR, f"{vname}_f{best_frame_id}.png")
    cv2.imwrite(out_path, frame)
    return vname


def main(max_players=3, workers=8):
    os.makedirs(OUT_DIR, exist_ok=True)

    files = [f.replace(".mp4", "") for f in os.listdir(GOOD_ROUNDS_DIR) if f.endswith(".mp4")]
    print(f"Total good quality rounds: {len(files)}", flush=True)

    items = [(vname, max_players) for vname in sorted(files)]
    sparse_videos = []
    n_total = len(items)

    with mp.Pool(workers) as pool:
        for i, result in enumerate(pool.imap_unordered(_process_one, items), 1):
            if result is not None:
                sparse_videos.append(result)
            if i % 500 == 0 or i == n_total:
                print(f"[{i}/{n_total}] sparse so far: {len(sparse_videos)}", flush=True)

    txt_path = os.path.join(OUT_DIR, "sparse_player_videos.txt")
    with open(txt_path, "w") as f:
        for v in sorted(sparse_videos):
            f.write(v + "\n")

    print(f"\nDone! {len(sparse_videos)} videos with <={max_players} active players "
          f"out of {n_total}", flush=True)
    print(f"Images + txt saved to: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max_players", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    main(max_players=args.max_players, workers=args.workers)
