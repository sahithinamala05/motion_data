"""Track the fixed table devices (discard tray / secondary tray / shoe+scanner)
in a live-dealer blackjack video with SAM 3, then label each track by its
fixed screen position. Outputs a browser-viewable H.264 video + summary JSON.

    python track_table_objects.py <video.mp4>
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
PROMPT = "transparent plastic box"
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

predictor = build_sam3_video_predictor()
resp = predictor.handle_request(dict(type="start_session", resource_path=VIDEO))
sid = resp["session_id"]
resp = predictor.handle_request(dict(
    type="add_prompt", session_id=sid, frame_index=0, text=PROMPT))
print(f"[info] frame0 detections: {len(resp['outputs']['out_obj_ids'])}", flush=True)

outputs_per_frame = {}
for r in predictor.handle_stream_request(dict(type="propagate_in_video", session_id=sid)):
    outputs_per_frame[r["frame_index"]] = r["outputs"]
predictor.handle_request(dict(type="close_session", session_id=sid))
print(f"[info] propagated over {len(outputs_per_frame)} frames", flush=True)


def box_xyxy(bx):
    bx = bx.tolist() if hasattr(bx, "tolist") else list(bx)
    x, y, w, h = [float(v) for v in bx]
    if max(x, y, w, h) <= 1.5:
        x, w = x * W, w * W
        y, h = y * H, h * H
    return [x, y, x + w, y + h]


# gather per-track box centers + persistence
centers = defaultdict(list)   # id -> [(cx, cy)]
boxes_by_frame = {}           # fidx -> [(id, xyxy)]
for fidx in sorted(outputs_per_frame):
    out = outputs_per_frame[fidx]
    ids = out["out_obj_ids"].tolist() if hasattr(out["out_obj_ids"], "tolist") else list(out["out_obj_ids"])
    bxs = out["out_boxes_xywh"]
    row = []
    for j, oid in enumerate(ids):
        b = box_xyxy(bxs[j])
        centers[int(oid)].append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
        row.append((int(oid), b))
    boxes_by_frame[fidx] = row

# keep the 3 most-persistent tracks (the fixed devices)
persistence = {oid: len(c) for oid, c in centers.items()}
top = sorted(persistence, key=lambda o: -persistence[o])[:3]
mean_c = {oid: (float(np.mean([c[0] for c in centers[oid]])),
                float(np.mean([c[1] for c in centers[oid]]))) for oid in top}

# assign roles by fixed screen position
role = {}
right = max(top, key=lambda o: mean_c[o][0])          # largest x -> right device
role[right] = "shoe/scanner"
left_ids = [o for o in top if o != right]
if len(left_ids) == 2:
    # of the two left devices, the lower one (larger y) is the discard tray
    lower = max(left_ids, key=lambda o: mean_c[o][1])
    upper = [o for o in left_ids if o != lower][0]
    role[lower] = "discard tray"
    role[upper] = "secondary tray"
elif len(left_ids) == 1:
    role[left_ids[0]] = "discard tray"

print("\n===== TABLE DEVICES =====")
for oid in top:
    cx, cy = mean_c[oid]
    print(f"  track id{oid:<3} -> {role.get(oid,'?'):<14} "
          f"center=({cx:.0f},{cy:.0f}) frames={persistence[oid]}")

summary = {
    "video": VIDEO, "prompt": PROMPT, "frames": len(frames),
    "fps": round(fps, 2), "resolution": f"{W}x{H}",
    "devices": [
        {"obj_id": int(oid), "role": role.get(oid, "object"),
         "center_xy": [round(mean_c[oid][0]), round(mean_c[oid][1])],
         "frames_present": persistence[oid]}
        for oid in top
    ],
}
with open(f"{OUT_DIR}/{stem}_devices.json", "w") as fh:
    json.dump(summary, fh, indent=2)

# render labeled video
ROLE_COLOR = {"discard tray": (0, 0, 255), "secondary tray": (0, 200, 255),
              "shoe/scanner": (0, 255, 0)}
tmp = f"{OUT_DIR}/{stem}_devices_tmp.mp4"
vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
for fidx, frame in enumerate(frames):
    img = frame.copy()
    for oid, b in boxes_by_frame.get(fidx, []):
        if oid not in role:
            continue
        x1, y1, x2, y2 = [int(v) for v in b]
        color = ROLE_COLOR.get(role[oid], (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = role[oid]
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    vw.write(img)
vw.release()
out_mp4 = f"{OUT_DIR}/{stem}_devices_h264.mp4"
os.system(f'ffmpeg -y -loglevel error -i "{tmp}" -c:v libx264 -pix_fmt yuv420p '
          f'-movflags +faststart "{out_mp4}" && rm -f "{tmp}"')
print(f"\n[done] labeled video: {out_mp4}")
print(f"[done] devices json:  {OUT_DIR}/{stem}_devices.json")
