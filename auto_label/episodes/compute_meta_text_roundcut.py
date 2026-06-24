"""
Compute meta_text using per-round card detection JSONLs from
card_detections_roundwise/ (roundcut-derived) instead of the 10s-chunk source.

Same Euclidean+angular-distance check (assign_cards_to_positions in
clustering_util.py) — only the data source differs:
  - Lookup is direct: frame_id in the JSONL is clip-relative, so we use
    segment['end_frame'] of "initial hands 1st" directly (no clip_start offset).
  - polygon_center comes nested ([[x,y]]) so we flatten before clustering.

Usage:
    python -m episodes.compute_meta_text_roundcut
or:
    python compute_meta_text_roundcut.py
"""
import os
import sys
import json
import glob
from collections import defaultdict
from tqdm import tqdm

# clustering_util lives in the sibling 10s-chunk dir
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "10s-chunk"))
from clustering_util import get_active_players  # noqa: E402

PRED_JSON_DIR = "/home/ubuntu/us-west-3-fs/sahithi/Ouput/remaining/predictions_json"
CARDS_DIR     = "/home/ubuntu/us-west-3-fs/sahithi/hard_action_annotations/card_detections_roundwise"
OUTPUT_DIR    = "/home/ubuntu/us-west-3-fs/sahithi/Ouput/remaining/predictions_json_with_meta_roundcut"

META_TEXT_LABELS = {"initial hands 1st", "initial hands 2nd", "discard"}
NEAREST_FRAME_TOLERANCE = 60   # frames (~2s @ 30 fps)


def load_detections_at_frame(jsonl_path: str, target_frame: int):
    """Return detections at `target_frame`, else nearest frame within ±tolerance."""
    by_frame = {}
    with open(jsonl_path) as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("detections"):
                by_frame[obj["frame_id"]] = obj["detections"]
    if not by_frame:
        return []
    if target_frame in by_frame:
        return by_frame[target_frame]
    nearest = min(by_frame.keys(), key=lambda f: abs(f - target_frame))
    return by_frame[nearest] if abs(nearest - target_frame) <= NEAREST_FRAME_TOLERANCE else []


def flatten_polygon_center(dets):
    """clustering_util expects polygon_center as [x,y]; JSONL stores [[x,y]]."""
    out = []
    for d in dets:
        pc = d.get("polygon_center")
        if isinstance(pc, list) and pc and isinstance(pc[0], list):
            d = {**d, "polygon_center": pc[0]}
        out.append(d)
    return out


def main():
    print("=" * 70)
    print("COMPUTE META_TEXT FROM ROUNDCUT CARD DETECTIONS")
    print("=" * 70)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pred_files = sorted(glob.glob(os.path.join(PRED_JSON_DIR, "*_predictions.json")))
    print(f"Prediction JSONs: {len(pred_files)}")
    print(f"Card-detection JSONLs: {len(glob.glob(os.path.join(CARDS_DIR, '*.jsonl')))}")

    stats = defaultdict(int)
    label_counts = defaultdict(int)

    for pred_path in tqdm(pred_files, desc="Processing"):
        round_name = os.path.basename(pred_path).replace("_predictions.json", "")
        jsonl_path = os.path.join(CARDS_DIR, f"{round_name}_card.jsonl")
        with open(pred_path) as f:
            pred = json.load(f)

        meta_text = ""
        if os.path.exists(jsonl_path):
            for seg in pred.get("timeline_segments", []):
                if "initial hands 1st" in seg.get("labels", []):
                    dets = load_detections_at_frame(jsonl_path, seg["end_frame"])
                    dets = flatten_polygon_center(dets)
                    meta_text = get_active_players(dets)
                    break
        else:
            stats["missing_jsonl"] += 1

        for seg in pred.get("timeline_segments", []):
            label = seg["labels"][0] if seg.get("labels") else ""
            if label in META_TEXT_LABELS:
                seg["meta_text"] = [meta_text]
                label_counts[label] += 1

        if meta_text:
            stats["files_with_meta"] += 1
        stats["total"] += 1

        out_path = os.path.join(OUTPUT_DIR, os.path.basename(pred_path))
        with open(out_path, "w") as f:
            json.dump(pred, f, indent=2)

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total processed:            {stats['total']}")
    print(f"Files with non-empty meta:  {stats['files_with_meta']}")
    print(f"Files missing card JSONL:   {stats['missing_jsonl']}")
    print("Segments updated per label:")
    for label in sorted(label_counts.keys()):
        print(f"  {label}: {label_counts[label]}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
