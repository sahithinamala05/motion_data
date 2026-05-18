"""Generate vis clips for all verified split segments (1606) from delivered JSONs."""

import json
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from process_split_segments_dino import visualize_success

SRC_DIR = "/home/ubuntu/us-west-3-fs/live_dealer_blackjack/action_annotation/split"
OUT_DIR = "/home/ubuntu/us-west-3-fs/sahithi/hard_action_annotations/full_run/split_vis"


def _kf_to_card(kf, idx):
    """Reconstruct a card dict from a keyframe entry at index idx."""
    return {
        "rank": kf["rank"][idx],
        "suit": kf["suit"][idx],
        "conf": kf["conf"][idx],
        "box": kf["card_box"][idx],
        "polygon_center": [kf["polygon_center"][idx]],
    }


def main():
    vis_segments = []
    for fname in sorted(os.listdir(SRC_DIR)):
        if not fname.endswith(".json"):
            continue
        vname = fname.replace("_annotations.json", "")
        with open(os.path.join(SRC_DIR, fname)) as f:
            data = json.load(f)

        for seg in data.get("timeline_segments", []):
            bboxes = seg.get("bounding_boxes", [])
            if not bboxes:
                continue
            kfs = bboxes[0].get("keyframes", [])
            if len(kfs) < 2:
                continue

            kf_start, kf_end = kfs[0], kfs[-1]
            vis_segments.append({
                "video_name_base": vname,
                "start_frame": seg["start_frame"],
                "end_frame": seg["end_frame"],
                "pair_rank": kf_start["rank"][0],
                "pair_cards_start": (_kf_to_card(kf_start, 0), _kf_to_card(kf_start, 1)),
                "pair_cards_end": (_kf_to_card(kf_end, 0), _kf_to_card(kf_end, 1)),
                "start_det_frame": kf_start["frame"],
                "end_det_frame": kf_end["frame"],
            })

    print(f"Found {len(vis_segments)} segments to visualize", flush=True)
    visualize_success(vis_segments, -1, OUT_DIR)


if __name__ == "__main__":
    main()
