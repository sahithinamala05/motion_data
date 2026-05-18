"""Run YOLO-OBB inference on a range of frames in each round video.

Input list: each line `video_path|start_frame|end_frame|orig_clip_name`.
Output: one annotated mp4 per input clip, containing only frames in [start, end].
"""

import argparse
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
        cv2.putText(frame, f"{SEAT_NAMES[int(cid)]} {c:.2f}", (cx - 25, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)


def process_clip(model, video_path, start, end, out_path, imgsz=1280, conf=0.3):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (IMG_W, IMG_H))
    n_done, n_seats_total = 0, 0
    fid = start - 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fid += 1
        if fid > end:
            break
        if (frame.shape[1], frame.shape[0]) != (IMG_W, IMG_H):
            frame = cv2.resize(frame, (IMG_W, IMG_H))
        r = model.predict(source=frame, imgsz=imgsz, conf=conf, verbose=False)[0]
        draw_obb_results(frame, r)
        if r.obb is not None:
            n_seats_total += len(r.obb)
        cv2.putText(frame, f"f={fid}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(frame)
        n_done += 1
    cap.release()
    writer.release()
    return {"frames": n_done, "total_seats": n_seats_total}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--list", required=True,
                   help="Pipe-separated lines: video_path|start|end|orig_name")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.3)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clips = [l.strip() for l in open(args.list) if l.strip()]
    print(f"Loaded {len(clips)} clips")
    model = YOLO(args.model)

    for i, line in enumerate(clips, 1):
        video, start_s, end_s, orig = line.split("|")
        start, end = int(start_s), int(end_s)
        out_name = orig.replace("_vis.mp4", "_obb.mp4")
        out_path = out_dir / out_name
        print(f"[{i}/{len(clips)}] {orig} (frames {start}-{end})", flush=True)
        stats = process_clip(model, video, start, end, str(out_path),
                             args.imgsz, args.conf)
        if stats is None:
            print("  failed to open")
            continue
        avg = stats["total_seats"] / max(1, stats["frames"])
        print(f"  {stats['frames']} frames, total_seats={stats['total_seats']}, "
              f"avg/frame={avg:.2f}  -> {out_path.name}", flush=True)


if __name__ == "__main__":
    main()
