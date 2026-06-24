#!/usr/bin/env python3
"""
Visualize initial-hand GROUPED RUNS on the source video.

Reads the per-video output of compute_initial_hand_cards.py (whose card_layout is a
list of run-length runs over the initial-hands segments) and renders the full deal
(initial hands 1st + 2nd) as a clip. For each frame we look up the run that contains
it and draw that run's representative cards in dealing order (players seat 1->7,
dealer last; dealer red, players green), with a header showing the current run/state.

Output clips are H.264 (re-encoded via ffmpeg) so they are viewable anywhere.
"""

import os
import json
import glob
import subprocess

import cv2

from compute_meta_text_and_export_v2 import parse_pred_json_filename

# ==================== CONFIGURATION ====================
INITIAL_HAND_DIR = "/home/ubuntu/us-west-3-fs/sahithi/initial_hand"
VIDEO_CUT_DIR = "/home/ubuntu/us-west-3-fs/live_dealer_blackjack/video_cut/batch_01"
VIS_OUT_DIR = "/home/ubuntu/us-west-3-fs/sahithi/initial_hand_vis"

N_EXAMPLES = 100
CHUNK = 300
REF_W, REF_H = 1920, 1080
DEFAULT_FPS = 25.0

DEALER_COLOR = (0, 0, 255)   # red (BGR)
PLAYER_COLOR = (0, 200, 0)   # green


def chunk_video_for_cstart(base_name, cstart):
    stem = f"{base_name}_{cstart // CHUNK:06d}_{cstart:06d}"
    return os.path.join(VIDEO_CUT_DIR, stem + ".mp4")


def gather_runs(data):
    """Return (runs, gstart, gend). Each run: (label, run_dict)."""
    runs = []
    gstart = gend = None
    for seg in data.get("initial_hands", []):
        gs = seg.get("global_start_frame")
        ge = seg.get("global_end_frame")
        if gs is None or ge is None:
            continue
        gstart = gs if gstart is None else min(gstart, gs)
        gend = ge if gend is None else max(gend, ge)
        for run in seg.get("card_layout", []):
            runs.append((seg["label"], run))
    runs.sort(key=lambda lr: lr[1]["start_frame"])
    return runs, gstart, gend


def run_at(runs, gframe):
    for i, (label, run) in enumerate(runs):
        if run["start_frame"] <= gframe <= run["end_frame"]:
            return i, label, run
    return None, None, None


def draw(frame, label, run, run_idx, n_runs):
    h, w = frame.shape[:2]
    sx, sy = w / REF_W, h / REF_H
    for c in run["cards"]:
        cx, cy = c["centroid"]
        px, py = int(cx * sx), int(cy * sy)
        color = DEALER_COLOR if c["role"] == "dealer" else PLAYER_COLOR
        tag = "D" if c["role"] == "dealer" else f"P{c['seat']}"
        cv2.circle(frame, (px, py), 11, color, 2)
        cv2.putText(frame, f"{c['order']}:{tag}", (px + 12, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    header = f"{label} | run {run_idx + 1}/{n_runs} | f{run['start_frame']}-{run['end_frame']} | n={run['n_cards']}"
    cv2.rectangle(frame, (0, 0), (w, 34), (0, 0, 0), -1)
    cv2.putText(frame, header, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def render(base_name, runs, gstart, gend, tmp_path):
    first_video = chunk_video_for_cstart(base_name, (gstart // CHUNK) * CHUNK)
    cap0 = cv2.VideoCapture(first_video)
    if not cap0.isOpened():
        return False
    fps = cap0.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    w = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap0.release()

    writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    cap = None
    cstart = -1
    wrote = 0
    for f in range(gstart, gend + 1):
        c = (f // CHUNK) * CHUNK
        if c != cstart:
            if cap is not None:
                cap.release()
            cap = cv2.VideoCapture(chunk_video_for_cstart(base_name, c))
            cstart = c
            if not cap.isOpened():
                cap = None
                continue
        if cap is None:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, f - cstart)
        ok, frame = cap.read()
        if not ok:
            continue
        idx, label, run = run_at(runs, f)
        if run is not None:
            draw(frame, label, run, idx, len(runs))
        writer.write(frame)
        wrote += 1
    if cap is not None:
        cap.release()
    writer.release()
    return wrote > 0


def to_h264(tmp_path, final_path):
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", tmp_path,
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         "-an", final_path],
        check=True,
    )
    os.remove(tmp_path)


def main():
    os.makedirs(VIS_OUT_DIR, exist_ok=True)
    for old in glob.glob(os.path.join(VIS_OUT_DIR, "*.mp4")) + glob.glob(os.path.join(VIS_OUT_DIR, "*.jpg")):
        os.remove(old)

    out_files = sorted(glob.glob(os.path.join(INITIAL_HAND_DIR, "*_predictions.json")))
    print(f"Output JSONs available: {len(out_files)}")

    made = 0
    for pred_path in out_files:
        if made >= N_EXAMPLES:
            break
        data = json.load(open(pred_path))
        base_name, _, _ = parse_pred_json_filename(pred_path)
        runs, gstart, gend = gather_runs(data)
        if not runs or gstart is None or not any(r["cards"] for _, r in runs):
            continue
        stem = os.path.basename(pred_path).replace("_predictions.json", "")
        final_path = os.path.join(VIS_OUT_DIR, f"{stem}__runs__{gstart}-{gend}.mp4")
        tmp_path = os.path.join(VIS_OUT_DIR, f".tmp_{stem}.mp4")
        if render(base_name, runs, gstart, gend, tmp_path):
            to_h264(tmp_path, final_path)
            made += 1
            print(f"[{made}] {os.path.basename(final_path)}  ({len(runs)} runs, {gend - gstart + 1} frames)")
        elif os.path.exists(tmp_path):
            os.remove(tmp_path)

    print(f"\nSaved {made} run-overlay clips to {VIS_OUT_DIR}")


if __name__ == "__main__":
    main()
