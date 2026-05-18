"""
Find roundcut videos where every active player has >=N cards and dealer has >=M.

Scans card detections, clusters into seats, saves a representative frame
for qualifying videos.

Usage:
    python find_crowded_player_videos.py [--min_player_cards N] [--min_dealer_cards N] [--workers N]
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
OUT_DIR = "/home/ubuntu/us-west-3-fs/sahithi/clus_anno/crowded_all"

IMG_W, IMG_H = 1280, 720


def _get_raw_video(date_time):
    base = date_time.replace("_", " ", 1)
    for name in (f"{base}.mp4", f"{date_time}.mp4"):
        path = os.path.join(RAW_VIDEO_DIR, name)
        if os.path.exists(path):
            return path
    return None


def _process_one(args):
    vname, min_player_cards, min_dealer_cards = args
    card_jsonl = os.path.join(CARDS_DIR, f"{vname}_card.jsonl")
    if not os.path.exists(card_jsonl):
        return None

    crowded_frames = []
    with open(card_jsonl) as f:
        for line in f:
            obj = json.loads(line)
            if not obj["detections"]:
                continue
            if len(obj["detections"]) >= 23:
                crowded_frames.append(obj["frame_id"])

    if not crowded_frames:
        return None

    parts = vname.split("_")
    date_time = parts[0] + "_" + parts[1]
    clip_start = int(parts[2])
    raw_path = _get_raw_video(date_time)
    if not raw_path:
        return None

    cap = cv2.VideoCapture(raw_path)
    for frame_id in crowded_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, (clip_start - 1) + frame_id)
        ret, frame = cap.read()
        if not ret:
            continue
        if frame.shape[1] != IMG_W or frame.shape[0] != IMG_H:
            frame = cv2.resize(frame, (IMG_W, IMG_H))
        out_path = os.path.join(OUT_DIR, f"{vname}_f{frame_id}.png")
        cv2.imwrite(out_path, frame)
    cap.release()
    return vname


def main(min_player_cards=3, min_dealer_cards=2, workers=8):
    os.makedirs(OUT_DIR, exist_ok=True)

    files = [f.replace(".mp4", "") for f in os.listdir(GOOD_ROUNDS_DIR) if f.endswith(".mp4")]
    print(f"Total good quality rounds: {len(files)}", flush=True)

    items = [(vname, min_player_cards, min_dealer_cards) for vname in sorted(files)]
    crowded_videos = []
    n_total = len(items)

    with mp.Pool(workers) as pool:
        for i, result in enumerate(pool.imap_unordered(_process_one, items), 1):
            if result is not None:
                crowded_videos.append(result)
            if i % 500 == 0 or i == n_total:
                print(f"[{i}/{n_total}] crowded so far: {len(crowded_videos)}", flush=True)

    txt_path = os.path.join(OUT_DIR, "crowded_player_videos.txt")
    with open(txt_path, "w") as f:
        for v in sorted(crowded_videos):
            f.write(v + "\n")

    print(f"\nDone! {len(crowded_videos)} crowded videos out of {n_total}", flush=True)
    print(f"Images + txt saved to: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min_player_cards", type=int, default=3)
    parser.add_argument("--min_dealer_cards", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    main(min_player_cards=args.min_player_cards,
         min_dealer_cards=args.min_dealer_cards, workers=args.workers)
