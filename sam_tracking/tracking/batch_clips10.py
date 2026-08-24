"""Batch SAM 3 combined tracking (playing cards + plastic-box devices) over the
pre-cut 10-second clips in video_cut/batch_01.

Loads the predictor ONCE, processes the first N clips (sorted) sequentially, and
saves the combined H.264 mp4 + a small combined.json per clip. Resumable (skips
clips whose mp4 already exists) and fault-tolerant (one bad clip won't kill the
batch).

    python batch_clips10.py [N] [START]
"""
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import glob
import json
import time
import traceback
from collections import defaultdict

import cv2
import numpy as np
import torch

from sam3.model_builder import build_sam3_video_predictor

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

SRC_DIR = "/home/ubuntu/us-west-3-fs/live_dealer_blackjack/video_cut/batch_01"
OUT_DIR = "/home/ubuntu/us-west-3-fs/sahithi/SAM_1000_rounds"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
START = int(sys.argv[2]) if len(sys.argv) > 2 else 0

os.makedirs(OUT_DIR, exist_ok=True)
CLIP_LIST = os.environ.get("CLIP_LIST")
if CLIP_LIST:
    print(f"[init] reading clip list {CLIP_LIST}", flush=True)
    with open(CLIP_LIST) as fh:
        videos = [ln.strip() for ln in fh if ln.strip()][START:START + N]
else:
    print("[init] listing clips...", flush=True)
    videos = sorted(glob.glob(os.path.join(SRC_DIR, "*.mp4")))[START:START + N]
print(f"[init] {len(videos)} clips to process -> {OUT_DIR}", flush=True)

print("[init] building SAM3 video predictor (downloads ckpt first run)...", flush=True)
predictor = build_sam3_video_predictor()

ROLE_COLOR = {"discard tray": (0, 0, 255), "secondary tray": (0, 200, 255),
              "shoe/scanner": (0, 255, 0)}
CARD_COLORS = [(255, 200, 0), (255, 0, 255), (200, 120, 255), (255, 128, 0),
               (0, 255, 255), (128, 255, 0), (255, 0, 128), (0, 180, 255)]


def box_xyxy(bx, W, H):
    bx = bx.tolist() if hasattr(bx, "tolist") else list(bx)
    x, y, w, h = [float(v) for v in bx]
    if max(x, y, w, h) <= 1.5:
        x, w = x * W, w * W
        y, h = y * H, h * H
    return [x, y, x + w, y + h]


def label(img, box, text, color, thick=2, fs=0.6):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
    cv2.rectangle(img, (x1, y1 - th - 5), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 2)


def run_prompt(sid, text, W, H):
    predictor.handle_request(dict(type="reset_session", session_id=sid))
    predictor.handle_request(dict(type="add_prompt", session_id=sid, frame_index=0, text=text))
    per_frame = {}
    for r in predictor.handle_stream_request(dict(type="propagate_in_video", session_id=sid)):
        out = r["outputs"]
        ids = out["out_obj_ids"].tolist() if hasattr(out["out_obj_ids"], "tolist") else list(out["out_obj_ids"])
        bxs = out["out_boxes_xywh"]
        per_frame[r["frame_index"]] = [(int(ids[j]), box_xyxy(bxs[j], W, H)) for j in range(len(ids))]
    return per_frame


def process(video):
    stem = os.path.splitext(os.path.basename(video))[0]
    out_mp4 = f"{OUT_DIR}/{stem}_combined_h264.mp4"
    if os.path.exists(out_mp4) and os.path.getsize(out_mp4) > 0:
        return "skip"

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()
    if not frames:
        return "empty"
    H, W = frames[0].shape[:2]

    resp = predictor.handle_request(dict(type="start_session", resource_path=video))
    sid = resp["session_id"]
    try:
        cards = run_prompt(sid, "playing card", W, H)
        devices = run_prompt(sid, "transparent plastic box", W, H)
    finally:
        predictor.handle_request(dict(type="close_session", session_id=sid))
        torch.cuda.empty_cache()

    centers = defaultdict(list)
    for fr in devices.values():
        for oid, b in fr:
            centers[oid].append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
    persistence = {o: len(c) for o, c in centers.items()}
    top = sorted(persistence, key=lambda o: -persistence[o])[:3]
    mean_c = {o: (float(np.mean([c[0] for c in centers[o]])),
                  float(np.mean([c[1] for c in centers[o]]))) for o in top}
    role = {}
    if top:
        right = max(top, key=lambda o: mean_c[o][0])
        role[right] = "shoe/scanner"
        left = [o for o in top if o != right]
        if len(left) == 2:
            lower = max(left, key=lambda o: mean_c[o][1])
            role[lower] = "discard tray"
            role[[o for o in left if o != lower][0]] = "secondary tray"
        elif len(left) == 1:
            role[left[0]] = "discard tray"

    n_card_tracks = len({oid for fr in cards.values() for oid, _ in fr})

    tmp = f"{OUT_DIR}/{stem}_combined_tmp.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for fidx, frame in enumerate(frames):
        img = frame.copy()
        for oid, b in devices.get(fidx, []):
            if oid in role:
                label(img, b, role[oid], ROLE_COLOR.get(role[oid], (255, 255, 255)), thick=3, fs=0.7)
        for oid, b in cards.get(fidx, []):
            label(img, b, f"id{oid}", CARD_COLORS[oid % len(CARD_COLORS)], thick=2, fs=0.6)
        vw.write(img)
    vw.release()
    rc = os.system(f'ffmpeg -y -loglevel error -i "{tmp}" -c:v libx264 -pix_fmt yuv420p '
                   f'-movflags +faststart "{out_mp4}"')
    if os.path.exists(tmp):
        os.remove(tmp)
    if rc != 0 or not os.path.exists(out_mp4):
        return "ffmpeg_fail"
    return f"ok cards={n_card_tracks} dev={len(role)}"


done = skipped = failed = 0
t0 = time.time()
for i, video in enumerate(videos):
    name = os.path.basename(video)
    try:
        t1 = time.time()
        status = process(video)
        dt = time.time() - t1
        if status == "skip":
            skipped += 1
            print(f"[{i+1}/{len(videos)}] SKIP {name}", flush=True)
        else:
            done += 1
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(videos) - i - 1) / rate if rate else 0
            print(f"[{i+1}/{len(videos)}] {status} {name} ({dt:.0f}s) "
                  f"| done={done} skip={skipped} fail={failed} | ETA {eta/3600:.1f}h", flush=True)
    except Exception:
        failed += 1
        print(f"[{i+1}/{len(videos)}] FAIL {name}\n{traceback.format_exc()}", flush=True)
        torch.cuda.empty_cache()

print(f"\n[batch done] ok={done} skipped={skipped} failed={failed} "
      f"in {(time.time()-t0)/3600:.2f}h -> {OUT_DIR}", flush=True)
