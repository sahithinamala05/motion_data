#!/usr/bin/env python3
"""
Convert the Label Studio "Initial Hands Adjustment" export into the
per-clip *_predictions.json files that
auto_label/10s-chunk/compute_initial_hand_cards.py and
compute_meta_text_and_export_v2.py consume.

Input  : one Label Studio JSON array (378 tasks), each task = one video clip
         with a single annotation holding
           - timelinelabels "initial hands 1st"/"initial hands 2nd" (frame ranges)
           - a "seats" textarea giving the active seats (e.g. "13")
Output : one file per clip named "<stem>_predictions.json", where <stem> is the
         mp4 basename "YYYY-MM-DD_HH-MM-SS_XXXXXX_YYYYYY". Each file:
           {
             "video_name": <stem>,
             "clip_start_1idx": <int>,            # from the filename
             "timeline_segments": [
               {"labels": ["initial hands 1st"],
                "start_frame": int, "end_frame": int,   # clip-relative (as labelled)
                "meta_text": ["13"]},                    # active seats
               ...
             ]
           }

Notes
-----
* The pipeline derives clip_start_1idx from the FILENAME, so the segment
  start_frame/end_frame are kept exactly as labelled (clip-relative); the
  pipeline adds (clip_start_1idx - 1) itself.
* A few clips contain two hands -> multiple "initial hands 1st"/"2nd" ranges.
  Every labelled range becomes its own timeline_segment.
* The single "seats" value is written as meta_text on every initial-hand
  segment in that clip (the pipeline likewise stamps one meta_text per file).
"""

import os
import re
import json
import argparse

IH_LABELS = {"initial hands 1st", "initial hands 2nd"}
STEM_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_(\d{6})_(\d{6})$")


def stem_from_video(video_uri: str) -> str:
    return video_uri.split("/")[-1].rsplit(".", 1)[0]


def convert_task(task: dict):
    """Return (stem, pred_dict) for one Label Studio task, or None to skip."""
    anns = task.get("annotations") or []
    if not anns:
        return None
    result = anns[0].get("result", [])

    stem = stem_from_video(task["data"]["video"])
    m = STEM_RE.match(stem)
    if not m:
        return None
    clip_start_1idx = int(m.group(2))

    # Active seats (single textarea per task).
    seats = ""
    for r in result:
        if r.get("type") == "textarea" and r.get("from_name") == "seats":
            seats = "".join(r["value"].get("text", [])).strip()
            break

    # One timeline_segment per labelled range.
    segments = []
    for r in result:
        if r.get("type") != "timelinelabels":
            continue
        label = r["value"]["timelinelabels"][0]
        if label not in IH_LABELS:
            continue
        for rng in r["value"]["ranges"]:
            segments.append(
                {
                    "labels": [label],
                    "start_frame": rng["start"],
                    "end_frame": rng["end"],
                    "meta_text": [seats],
                }
            )

    if not segments:
        return None

    segments.sort(key=lambda s: (s["start_frame"], s["labels"][0]))
    pred = {
        "video_name": stem,
        "clip_start_1idx": clip_start_1idx,
        "timeline_segments": segments,
    }
    return stem, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export", help="Label Studio export JSON")
    ap.add_argument(
        "-o", "--out-dir",
        default="/home/ubuntu/sahithi/initial_hand_adj_predictions_json",
        help="output directory for *_predictions.json files",
    )
    args = ap.parse_args()

    with open(args.export) as f:
        tasks = json.load(f)

    os.makedirs(args.out_dir, exist_ok=True)

    written, skipped, n_segments = 0, 0, 0
    for task in tasks:
        out = convert_task(task)
        if out is None:
            skipped += 1
            continue
        stem, pred = out
        n_segments += len(pred["timeline_segments"])
        with open(os.path.join(args.out_dir, f"{stem}_predictions.json"), "w") as f:
            json.dump(pred, f, indent=2)
        written += 1

    print(f"tasks in export      : {len(tasks)}")
    print(f"prediction files out : {written}")
    print(f"tasks skipped        : {skipped}")
    print(f"timeline_segments    : {n_segments}")
    print(f"output dir           : {args.out_dir}")


if __name__ == "__main__":
    main()
