"""Render adjustment videos for every round flagged by find_adjustment_rounds.py.

Parses the rank list out of the most-recent scan log and calls render() on each
round directly (no per-call Python startup). Skips rounds whose output mp4 is
already present, so it's safely resumable.
"""
import csv
import os
import re
import sys

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from viz_initial_hands_adjustment import (
    DEFAULT_PRED_DIR, DEFAULT_CARDS_DIR, DEFAULT_VIDEO_DIR, DEFAULT_OUTPUT_DIR,
    render,
)

CSV_PATH = os.path.join(DEFAULT_OUTPUT_DIR, "adjustments_summary.csv")

MANIFEST = "/home/ubuntu/sahithi/FACT_actseg/adjustment_rounds.txt"


def round_names_from_manifest(path):
    names = []
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                names.append(parts[2])
    return names


def main():
    rounds = round_names_from_manifest(MANIFEST)
    print(f"will render {len(rounds)} rounds → {DEFAULT_OUTPUT_DIR}")
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    failed = []
    skipped = 0
    rows = []
    for r in tqdm(rounds, desc="render"):
        out = os.path.join(DEFAULT_OUTPUT_DIR, f"{r}_adjust.mp4")
        if os.path.exists(out):
            skipped += 1
            continue
        try:
            events = render(r, DEFAULT_PRED_DIR, DEFAULT_CARDS_DIR, DEFAULT_VIDEO_DIR, DEFAULT_OUTPUT_DIR) or []
            for ev in events:
                rows.append({
                    "round":         r,
                    "player":        ev["player"],
                    "card":          ev["card"],
                    "position":      ev["position"],
                    "stable_x":      round(ev["stable_xy"][0], 1),
                    "stable_y":      round(ev["stable_xy"][1], 1),
                    "stable_frame":  ev["stable_frame"],
                    "final_x":       round(ev["final_xy"][0], 1),
                    "final_y":       round(ev["final_xy"][1], 1),
                    "final_frame":   ev["final_frame"],
                    "delta_px":      round(ev["delta_px"], 1),
                    "adj_frame":     ev["adj_frame"],
                    "segment_start": ev["segment"][0],
                    "segment_end":   ev["segment"][1],
                })
        except Exception as e:
            print(f"!! {r}: {e}")
            failed.append((r, str(e)))

    # write CSV summary
    if rows:
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nsummary CSV: {CSV_PATH}  ({len(rows)} events)")

    print()
    print(f"rendered: {len(rounds) - skipped - len(failed)}")
    print(f"skipped (already done): {skipped}")
    print(f"failed: {len(failed)}")
    for r, e in failed[:20]:
        print(f"  {r}: {e}")


if __name__ == "__main__":
    main()
