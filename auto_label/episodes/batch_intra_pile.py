"""Run the intra-pile labeling pipeline on a batch of rounds.

Loads the YOLO model once, then iterates through round names. Saves JSON per
round and aggregates summary stats across the batch.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, "/home/ubuntu/sahithi/motion-data-process/auto_label/episodes")
from intra_pile_util import label_round
from gen_intra_pile_labels import (
    DEFAULT_MODEL, DEFAULT_CARDS_DIR, DEFAULT_VIDEO_DIR, DEFAULT_OUT_DIR,
    SEAT_NAMES,
    pick_rep_frame_by_detections, extract_video_frame_from_round,
    round_video_path, run_seat_obb,
    draw_raw_card_detections, draw_seat_obbs, draw_card_tracks,
    draw_fan_vectors, make_round_video,
)


def process_one(round_name, model, cards_dir, video_dir, out_dir,
                src_w=1280, src_h=720, img_w=1280, img_h=720, top_k=10,
                make_jpg=False, make_video=False):
    cards_path = Path(cards_dir) / f"{round_name}_card.jsonl"
    if not cards_path.exists():
        return {"round": round_name, "status": "missing_cards"}
    try:
        video = round_video_path(round_name, video_dir)
    except FileNotFoundError:
        return {"round": round_name, "status": "missing_video"}

    frames = [json.loads(l) for l in open(cards_path)]

    # Pick rep frame: max-detection candidates ranked by yolo seat count
    candidates = pick_rep_frame_by_detections(frames, top_k=top_k)
    best = None
    for fid, n in candidates:
        img = extract_video_frame_from_round(video, fid, img_w, img_h)
        if img is None:
            continue
        sobbs, _ = run_seat_obb(model, img)
        if best is None or len(sobbs) > best[0]:
            best = (len(sobbs), fid, img, sobbs)
            if len(sobbs) >= 8:
                break
    if best is None or len(best[3]) == 0:
        return {"round": round_name, "status": "no_seats_found"}
    rep_fid, rep_img, seat_obbs = best[1], best[2], best[3]

    tracks = label_round(frames, seat_obbs, src_w, src_h, img_w, img_h,
                         rep_fid=rep_fid)

    # Summarize
    by_seat = {}
    doubles = 0
    splits = 0
    unassigned = 0
    for t in tracks:
        s = t.get("seat")
        if s is None:
            unassigned += 1
            continue
        by_seat[s] = by_seat.get(s, 0) + 1
        if t.get("double"): doubles += 1
        if t.get("split_pile"): splits += 1

    # Serialize
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    serial = []
    for t in tracks:
        serial.append({
            "track_id": t["track_id"],
            "rank": t["rank"], "suit": t["suit"],
            "seat": t.get("seat"),
            "seat_name": SEAT_NAMES[t["seat"]] if t.get("seat") is not None else None,
            "seat_via": t.get("seat_via"),
            "ordinal": t.get("ordinal"),
            "double": t.get("double", False),
            "split_pile": t.get("split_pile", 0),
            "first_frame": t["first_frame"],
            "last_frame": t["last_frame"],
            "first_center": t["first_center"],
            "last_center": t["centers"][-1],
            "n_observations": len(t["centers"]),
            "mean_conf": float(np.mean(t["confs"])),
        })
    with open(out_dir / f"{round_name}_per_card.json", "w") as f:
        json.dump({
            "round": round_name,
            "representative_frame": f"round_fid={rep_fid}",
            "seat_obbs": seat_obbs,
            "tracks": serial,
        }, f)

    if make_jpg:
        from intra_pile_util import dedup_frame_detections
        rep_dets = next((fr.get("detections") or [] for fr in frames
                         if fr.get("frame_id") == rep_fid), [])
        rep_dets = dedup_frame_detections(rep_dets)
        img = rep_img.copy()
        draw_raw_card_detections(img, rep_dets, src_w, src_h, img_w, img_h)
        draw_seat_obbs(img, seat_obbs)
        draw_card_tracks(img, tracks, seat_obbs, src_w, src_h, img_w, img_h,
                         current_fid=rep_fid, frame_detections=rep_dets)
        draw_fan_vectors(img, tracks, seat_obbs, rep_dets)
        cv2.imwrite(str(out_dir / f"{round_name}_vis.jpg"), img)

    if make_video:
        make_round_video(round_name, frames, tracks, seat_obbs, video_dir,
                         str(out_dir / f"{round_name}_vis.mp4"),
                         src_w, src_h, img_w, img_h)

    return {
        "round": round_name, "status": "ok",
        "rep_fid": rep_fid,
        "n_seats": len(seat_obbs),
        "n_tracks": len(tracks),
        "n_unassigned": unassigned,
        "n_doubles": doubles,
        "n_splits": splits,
        "by_seat": by_seat,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rounds_file", required=True, help="Newline-separated round names")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--cards_dir", default=DEFAULT_CARDS_DIR)
    p.add_argument("--video_dir", default=DEFAULT_VIDEO_DIR)
    p.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--summary_json", default=None)
    p.add_argument("--jpg", action="store_true", help="Also save annotated JPG per round")
    p.add_argument("--video", action="store_true", help="Also save annotated MP4 per round")
    args = p.parse_args()

    rounds = [l.strip() for l in open(args.rounds_file) if l.strip()]
    print(f"Loaded {len(rounds)} rounds from {args.rounds_file}")

    model = YOLO(args.model)
    print("Model loaded")

    results = []
    t0 = time.time()
    for i, r in enumerate(rounds, 1):
        res = process_one(r, model, args.cards_dir, args.video_dir, args.out_dir,
                          make_jpg=args.jpg, make_video=args.video)
        results.append(res)
        if i % 10 == 0 or i == len(rounds):
            elapsed = time.time() - t0
            ok = sum(1 for x in results if x["status"] == "ok")
            print(f"  [{i}/{len(rounds)}] ok={ok}  elapsed={elapsed:.0f}s", flush=True)

    # Aggregate
    ok_results = [r for r in results if r["status"] == "ok"]
    n_ok = len(ok_results)
    n_total = len(results)
    statuses = {}
    for r in results:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    print()
    print(f"Processed {n_total} rounds. Status breakdown: {statuses}")

    if n_ok > 0:
        n_seats = [r["n_seats"] for r in ok_results]
        n_tracks = [r["n_tracks"] for r in ok_results]
        n_unassigned = [r["n_unassigned"] for r in ok_results]
        any_splits = sum(1 for r in ok_results if r["n_splits"] > 0)
        any_doubles = sum(1 for r in ok_results if r["n_doubles"] > 0)
        print(f"Seats found  : mean={np.mean(n_seats):.2f}  median={np.median(n_seats):.0f}  min={min(n_seats)}  max={max(n_seats)}")
        print(f"Tracks       : mean={np.mean(n_tracks):.2f}  median={np.median(n_tracks):.0f}  min={min(n_tracks)}  max={max(n_tracks)}")
        print(f"Unassigned   : mean={np.mean(n_unassigned):.2f}  any-unassigned-rounds={sum(1 for u in n_unassigned if u > 0)}/{n_ok}")
        print(f"Doubles flagged in: {any_doubles}/{n_ok} rounds")
        print(f"Splits  flagged in: {any_splits}/{n_ok} rounds")

    if args.summary_json:
        with open(args.summary_json, "w") as f:
            json.dump({"results": results, "statuses": statuses}, f, indent=2)
        print(f"Wrote {args.summary_json}")


if __name__ == "__main__":
    main()
