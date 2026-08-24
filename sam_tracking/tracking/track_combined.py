"""Combined SAM 3 tracking: playing cards + table devices in ONE video.

Runs two prompt passes in a single session (model loads once), then renders a
single browser-viewable H.264 video with card track IDs and the 3 position-
labeled devices (discard tray / secondary tray / shoe-scanner).

    python track_combined.py <video.mp4>
"""
import os
import sys
import json
from collections import defaultdict

import cv2
import numpy as np
import torch

from sam3.model_builder import build_sam3_video_predictor

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

VIDEO = sys.argv[1]
OUT_DIR = "/home/ubuntu/sahithi/sam3_out"
os.makedirs(OUT_DIR, exist_ok=True)
stem = os.path.splitext(os.path.basename(VIDEO))[0]

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
frames = []
while True:
    ret, f = cap.read()
    if not ret:
        break
    frames.append(f)
cap.release()
H, W = frames[0].shape[:2]
print(f"[info] {len(frames)} frames @ {fps:.1f} fps, {W}x{H}", flush=True)


def box_xyxy(bx):
    bx = bx.tolist() if hasattr(bx, "tolist") else list(bx)
    x, y, w, h = [float(v) for v in bx]
    if max(x, y, w, h) <= 1.5:
        x, w = x * W, w * W
        y, h = y * H, h * H
    return [x, y, x + w, y + h]


predictor = build_sam3_video_predictor()
resp = predictor.handle_request(dict(type="start_session", resource_path=VIDEO))
sid = resp["session_id"]


def run_prompt(text):
    predictor.handle_request(dict(type="reset_session", session_id=sid))
    predictor.handle_request(dict(type="add_prompt", session_id=sid,
                                  frame_index=0, text=text))
    per_frame = {}
    for r in predictor.handle_stream_request(dict(type="propagate_in_video", session_id=sid)):
        out = r["outputs"]
        ids = out["out_obj_ids"].tolist() if hasattr(out["out_obj_ids"], "tolist") else list(out["out_obj_ids"])
        bxs = out["out_boxes_xywh"]
        per_frame[r["frame_index"]] = [(int(ids[j]), box_xyxy(bxs[j])) for j in range(len(ids))]
    return per_frame


print("[info] pass 1/2: playing card", flush=True)
cards = run_prompt("playing card")
print("[info] pass 2/2: transparent plastic box", flush=True)
devices = run_prompt("transparent plastic box")
predictor.handle_request(dict(type="close_session", session_id=sid))

# --- assign device roles by fixed screen position ---
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
print(f"\n[info] cards: {n_card_tracks} tracks | devices: "
      f"{[role.get(o,'?') for o in top]}", flush=True)

summary = {
    "video": VIDEO, "frames": len(frames), "fps": round(fps, 2),
    "resolution": f"{W}x{H}", "card_tracks": n_card_tracks,
    "devices": [{"obj_id": int(o), "role": role.get(o, "object"),
                 "center_xy": [round(mean_c[o][0]), round(mean_c[o][1])],
                 "frames_present": persistence[o]} for o in top],
}
with open(f"{OUT_DIR}/{stem}_combined.json", "w") as fh:
    json.dump(summary, fh, indent=2)

# --- render combined video ---
ROLE_COLOR = {"discard tray": (0, 0, 255), "secondary tray": (0, 200, 255),
              "shoe/scanner": (0, 255, 0)}
CARD_COLORS = [(255, 200, 0), (255, 0, 255), (200, 120, 255), (255, 128, 0),
               (0, 255, 255), (128, 255, 0), (255, 0, 128), (0, 180, 255)]


def label(img, box, text, color, thick=2, fs=0.55):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
    cv2.rectangle(img, (x1, y1 - th - 5), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 2)


tmp = f"{OUT_DIR}/{stem}_combined_tmp.mp4"
vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
for fidx, frame in enumerate(frames):
    img = frame.copy()
    # devices first (thick), so cards draw on top
    for oid, b in devices.get(fidx, []):
        if oid in role:
            label(img, b, role[oid], ROLE_COLOR.get(role[oid], (255, 255, 255)), thick=3, fs=0.6)
    for oid, b in cards.get(fidx, []):
        label(img, b, f"id{oid}", CARD_COLORS[oid % len(CARD_COLORS)], thick=2, fs=0.5)
    vw.write(img)
vw.release()
out_mp4 = f"{OUT_DIR}/{stem}_combined_h264.mp4"
os.system(f'ffmpeg -y -loglevel error -i "{tmp}" -c:v libx264 -pix_fmt yuv420p '
          f'-movflags +faststart "{out_mp4}" && rm -f "{tmp}"')
print(f"\n[done] combined video: {out_mp4}")
print(f"[done] combined json:  {OUT_DIR}/{stem}_combined.json")
