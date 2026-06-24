"""Scan all easy-action rounds and list those with intra-seat card adjustments.

Uses the same detection as viz_initial_hands_adjustment.detect_adjustment_clips
(per-seat clustering, stable-vs-final, y > PLAYER_MIN_Y_PX). Prints a ranked
summary so you can pick a round to visualize.
"""
import os
import json
import glob

from tqdm import tqdm

from viz_initial_hands_adjustment import (
    DEFAULT_PRED_DIR, DEFAULT_CARDS_DIR,
    LABELS_TO_RENDER, ADJUST_THRESH_PX, PLAYER_MIN_Y_PX,
    load_card_jsonl, detect_adjustment_clips,
)


def main():
    pred_files = sorted(glob.glob(os.path.join(DEFAULT_PRED_DIR, "*_predictions.json")))
    print(f"scanning {len(pred_files)} rounds  (player zone y > {PLAYER_MIN_Y_PX}, "
          f"per-seat, threshold {ADJUST_THRESH_PX}px)")

    results = []   # (max_delta, n_events, round_name, events)
    n_missing_cards = 0
    n_no_segments  = 0

    for pred_path in tqdm(pred_files, desc="scan"):
        round_name = os.path.basename(pred_path).replace("_predictions.json", "")
        cards_path = os.path.join(DEFAULT_CARDS_DIR, f"{round_name}_card.jsonl")
        if not os.path.exists(cards_path):
            n_missing_cards += 1
            continue

        with open(pred_path) as f:
            pred = json.load(f)
        segs = [s for s in pred["timeline_segments"]
                if s.get("labels") and s["labels"][0] in LABELS_TO_RENDER]
        if not segs:
            n_no_segments += 1
            continue

        frame_lo = min(s["start_frame"] for s in segs)
        frame_hi = max(s["end_frame"]   for s in segs)
        cards_by_frame = load_card_jsonl(cards_path)
        _, events = detect_adjustment_clips(cards_by_frame, frame_lo, frame_hi, 30.0, segments=segs)
        if not events:
            continue

        max_delta = max(e["delta_px"] for e in events)
        results.append((max_delta, len(events), round_name, events))

    results.sort(key=lambda r: -r[0])

    manifest_path = "/home/ubuntu/sahithi/FACT_actseg/adjustment_rounds.txt"
    with open(manifest_path, "w") as f:
        for max_delta, n, name, _ in results:
            f.write(f"{max_delta:.1f}\t{n}\t{name}\n")
    print()
    print(f"full manifest: {manifest_path}  ({len(results)} rounds)")

    print(f"rounds with adjustments: {len(results)} / {len(pred_files) - n_missing_cards}")
    print(f"missing card JSONL: {n_missing_cards}   no initial-hands segments: {n_no_segments}")
    print()
    print("top 30 by max delta_px:")
    print(f"  {'max_delta':>9}  {'#evt':>4}  round")
    for max_delta, n, name, _ in results[:30]:
        print(f"  {max_delta:9.1f}  {n:4d}  {name}")


if __name__ == "__main__":
    main()
