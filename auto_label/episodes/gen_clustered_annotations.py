"""
Run card-to-seat clustering on every PNG in clus_anno/to_annotate and save a
single JSON of seat-level annotations (axis-aligned bbox per seat).

Output schema:
{
  "image_size": {"width": 1280, "height": 720},
  "classes": ["Dealer", "P1", "P2", "P3", "P4", "P5", "P6", "P7"],
  "annotations": [
    {
      "image": "<png_name>",
      "boxes": [
        {"class": "Dealer", "class_id": 0,
         "x1": int, "y1": int, "x2": int, "y2": int}
      ]
    }
  ]
}
"""

import argparse
import json
import os
import sys

sys.path.insert(0, "/home/ubuntu/sahithi/motion-data-process/auto_label/10s-chunk")
from clustering_util import assign_cards_to_positions

CARDS_DIR = "/home/ubuntu/us-west-3-fs/sahithi/hard_action_annotations/card_detections_roundwise"
INPUT_DIR = "/home/ubuntu/us-west-3-fs/sahithi/clus_anno/to_annotate"
OUTPUT_JSON = "/home/ubuntu/us-west-3-fs/sahithi/clus_anno/to_annotate_clustered_annotations.json"

SRC_W, SRC_H = 1920, 1080
IMG_W, IMG_H = 1280, 720
SCALE = IMG_W / SRC_W
PAD = 5
SEAT_NAMES = ["Dealer", "P1", "P2", "P3", "P4", "P5", "P6", "P7"]


def load_card_dets(card_jsonl, frame_id):
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
    boxes = [[int(v * SCALE) for v in c["box"]] for c in cards]
    x1 = max(0, min(b[0] for b in boxes) - PAD)
    y1 = max(0, min(b[1] for b in boxes) - PAD)
    x2 = min(IMG_W, max(b[2] for b in boxes) + PAD)
    y2 = min(IMG_H, max(b[3] for b in boxes) + PAD)
    return x1, y1, x2, y2


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
    boxes = []
    for seat in range(8):
        cards = clusters.get(seat, [])
        if not cards:
            continue
        x1, y1, x2, y2 = union_box_px(cards)
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append({
            "class": SEAT_NAMES[seat],
            "class_id": seat,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        })

    if not boxes:
        return None, "no_clusters"
    return {"image": png_name, "boxes": boxes}, "ok"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", default=INPUT_DIR)
    parser.add_argument("--output_json", default=OUTPUT_JSON)
    args = parser.parse_args()

    pngs = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".png"))
    print(f"Found {len(pngs)} PNGs in {args.input_dir}")

    annotations = []
    stats = {"ok": 0, "no_card_jsonl": 0, "no_dets": 0, "no_clusters": 0,
             "bad_filename": 0}
    for i, name in enumerate(pngs, 1):
        ann, status = process_one(name)
        stats[status] = stats.get(status, 0) + 1
        if ann is not None:
            annotations.append(ann)
        if i % 100 == 0 or i == len(pngs):
            print(f"  [{i}/{len(pngs)}] " + " ".join(f"{k}={v}" for k, v in stats.items()),
                  flush=True)

    out = {
        "image_size": {"width": IMG_W, "height": IMG_H},
        "classes": SEAT_NAMES,
        "annotations": annotations,
    }
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {len(annotations)} annotated images to {args.output_json}")


if __name__ == "__main__":
    main()
