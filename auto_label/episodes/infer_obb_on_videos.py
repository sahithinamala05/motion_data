"""Run YOLO-OBB seat detection on every frame of a set of videos and write
the annotated mp4 alongside.

Loads the model once, iterates videos sequentially.
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, "/home/ubuntu/sahithi/motion-data-process/auto_label/episodes")
from gen_intra_pile_labels import SEAT_NAMES, SEAT_COLORS

DEFAULT_MODEL = "/home/ubuntu/us-west-3-fs/sahithi/yolo_obb_runs/human_v2/weights/best.pt"
IMG_W, IMG_H = 1280, 720


def draw_obb_results(frame, result):
    """Draw OBB polygons + labels on `frame` in-place."""
    if result.obb is None or len(result.obb) == 0:
        return
    polys = result.obb.xyxyxyxy.cpu().numpy()
    cls_ids = result.obb.cls.cpu().numpy().astype(int)
    confs = result.obb.conf.cpu().numpy()
    for poly, cid, c in zip(polys, cls_ids, confs):
        col = SEAT_COLORS[int(cid) % len(SEAT_COLORS)]
        pts = poly.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], True, col, 2)
        cx, cy = np.mean(poly, axis=0).astype(int)
        label = f"{SEAT_NAMES[int(cid)]} {c:.2f}"
        cv2.putText(frame, label, (cx - 25, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)


def process_video(model, in_path, out_path, imgsz=1280, conf=0.3):
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_size = (IMG_W, IMG_H)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, out_size)
    n_done, n_seats_total = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if (frame.shape[1], frame.shape[0]) != out_size:
            frame = cv2.resize(frame, out_size)
        r = model.predict(source=frame, imgsz=imgsz, conf=conf, verbose=False)[0]
        draw_obb_results(frame, r)
        if r.obb is not None:
            n_seats_total += len(r.obb)
        cv2.putText(frame, f"f={n_done}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(frame)
        n_done += 1
    cap.release()
    writer.release()
    return {
        "n_frames": n_done, "n_seats_total": n_seats_total,
        "avg_seats_per_frame": n_seats_total / max(1, n_done),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--videos_list", required=True,
                   help="Newline-separated list of absolute video paths")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.3)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = [l.strip() for l in open(args.videos_list) if l.strip()]
    print(f"Loaded {len(videos)} videos")

    model = YOLO(args.model)
    print(f"Model loaded: {args.model}")

    for i, v in enumerate(videos, 1):
        name = Path(v).stem
        out_path = out_dir / f"{name}_obb.mp4"
        print(f"[{i}/{len(videos)}] {name}", flush=True)
        stats = process_video(model, v, str(out_path), args.imgsz, args.conf)
        if stats is None:
            print(f"  failed to open")
            continue
        print(f"  frames={stats['n_frames']}  total_seats={stats['n_seats_total']}  "
              f"avg/frame={stats['avg_seats_per_frame']:.2f}  -> {out_path.name}",
              flush=True)


if __name__ == "__main__":
    main()
