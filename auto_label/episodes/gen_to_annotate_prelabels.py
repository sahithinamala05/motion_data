"""
Generate Label Studio prelabels JSON for images in clus_anno/to_annotate.

For each PNG, runs card-to-seat clustering and emits one rectanglelabels prediction
per non-empty seat (Dealer + P1-P7), matching the schema used in
clus_anno/crowded_250_prelabels.json.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/home/ubuntu/sahithi/motion-data-process/auto_label/10s-chunk")
from clustering_util import assign_cards_to_positions

CARDS_DIR = "/home/ubuntu/us-west-3-fs/sahithi/hard_action_annotations/card_detections_roundwise"
INPUT_DIR = "/home/ubuntu/us-west-3-fs/sahithi/clus_anno/to_annotate"
OUTPUT_JSON = "/home/ubuntu/us-west-3-fs/sahithi/clus_anno/to_annotate_prelabels.json"
S3_PREFIX = "s3://valkaai-research/projects/livedealer/dataset/processed/Clustering_annotation_data"

SRC_W, SRC_H = 1920, 1080
IMG_W, IMG_H = 1280, 720
SCALE = IMG_W / SRC_W
PAD = 5
SEAT_NAMES = ["Dealer", "P1", "P2", "P3", "P4", "P5", "P6", "P7"]


def load_card_dets(card_jsonl, frame_id):
    """Find detections from the row whose frame_id is closest to `frame_id`."""
    dets = None
    best_diff = 999
    with open(card_jsonl) as f:
        for line in f:
            obj = json.loads(line)
            diff = abs(obj["frame_id"] - frame_id)
            if diff < best_diff and obj["detections"]:
                best_diff = diff
                dets = obj["detections"]
            if obj["frame_id"] > frame_id + 10:
                break
    if dets:
        for d in dets:
            if isinstance(d["polygon_center"][0], list):
                d["polygon_center"] = d["polygon_center"][0]
    return dets


def union_box_px(cards):
    """Union of card boxes in 1280x720 px space, with pad."""
    boxes = [[int(v * SCALE) for v in c["box"]] for c in cards]
    x1 = max(0, min(b[0] for b in boxes) - PAD)
    y1 = max(0, min(b[1] for b in boxes) - PAD)
    x2 = min(IMG_W, max(b[2] for b in boxes) + PAD)
    y2 = min(IMG_H, max(b[3] for b in boxes) + PAD)
    return x1, y1, x2, y2


def make_result(png_name, idx, x1, y1, x2, y2, label):
    return {
        "id": f"{png_name}_{idx}",
        "type": "rectanglelabels",
        "from_name": "label",
        "to_name": "image",
        "original_width": IMG_W,
        "original_height": IMG_H,
        "value": {
            "x": x1 / IMG_W * 100,
            "y": y1 / IMG_H * 100,
            "width": (x2 - x1) / IMG_W * 100,
            "height": (y2 - y1) / IMG_H * 100,
            "rectanglelabels": [label],
        },
    }


def process_one(png_name):
    base = png_name.replace(".png", "")
    parts = base.rsplit("_f", 1)
    if len(parts) != 2:
        return None, "bad_filename"
    vname, frame_id_str = parts
    try:
        frame_id = int(frame_id_str)
    except ValueError:
        return None, "bad_filename"

    card_jsonl = os.path.join(CARDS_DIR, f"{vname}_card.jsonl")
    if not os.path.exists(card_jsonl):
        return None, "no_card_jsonl"

    dets = load_card_dets(card_jsonl, frame_id)
    if not dets:
        return None, "no_dets"

    clusters = assign_cards_to_positions(dets)
    results = []
    idx = 0
    for seat in range(8):
        cards = clusters.get(seat, [])
        if not cards:
            continue
        x1, y1, x2, y2 = union_box_px(cards)
        if x2 <= x1 or y2 <= y1:
            continue
        results.append(make_result(png_name, idx, x1, y1, x2, y2, SEAT_NAMES[seat]))
        idx += 1

    if not results:
        return None, "no_clusters"

    task = {
        "data": {"image": f"{S3_PREFIX}/{png_name}"},
        "predictions": [
            {"model_version": "clustering_v1", "result": results}
        ],
    }
    return task, "ok"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", default=INPUT_DIR)
    parser.add_argument("--output_json", default=OUTPUT_JSON)
    args = parser.parse_args()

    pngs = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".png"))
    print(f"Found {len(pngs)} PNGs in {args.input_dir}")

    tasks = []
    stats = {"ok": 0, "no_card_jsonl": 0, "no_dets": 0, "no_clusters": 0,
             "bad_filename": 0}
    for i, name in enumerate(pngs, 1):
        task, status = process_one(name)
        stats[status] = stats.get(status, 0) + 1
        if task is not None:
            tasks.append(task)
        if i % 100 == 0 or i == len(pngs):
            print(f"  [{i}/{len(pngs)}] ok={stats['ok']} "
                  f"no_jsonl={stats['no_card_jsonl']} "
                  f"no_dets={stats['no_dets']} "
                  f"no_clusters={stats['no_clusters']} "
                  f"bad={stats['bad_filename']}", flush=True)

    with open(args.output_json, "w") as f:
        json.dump(tasks, f, indent=2)

    print(f"\nWrote {len(tasks)} tasks to {args.output_json}")


if __name__ == "__main__":
    main()
