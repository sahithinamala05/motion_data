"""Render overlay mp4s for a random sample of roundcut videos using the
detections in roundcut_detections.jsonl. Boxes are static (one detection
per video on the middle frame) so we draw the same overlay on every frame.
"""
import argparse
import json
import random
import subprocess
from pathlib import Path

import cv2

ROOT = Path('/home/ubuntu/sahithi/motion-data-process/auto_label/yolo_table_objects')
DETS = ROOT / 'roundcut_detections.jsonl'
SRC_DIR = Path('/home/ubuntu/us-west-3-fs/live_dealer_blackjack/roundcut/good_quality_rounds')
OUT_DIR = ROOT / 'vis_roundcut'
FFMPEG = '/usr/bin/ffmpeg'

# BGR per class
COLORS = {
    'discard_tray': (0, 215, 255),   # gold
    'shoe':         (0, 255, 0),     # green
    'spare_shoe':   (255, 128, 0),   # cyan-blue
    'scanner':      (0, 0, 255),     # red
}


def draw_boxes(frame, dets):
    for d in dets:
        x1, y1, x2, y2 = map(int, d['bbox_xyxy'])
        color = COLORS.get(d['class'], (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{d['class']} {d['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ytxt = max(0, y1 - 5)
        cv2.rectangle(frame, (x1, ytxt - th - 4), (x1 + tw + 4, ytxt), color, -1)
        cv2.putText(frame, label, (x1 + 2, ytxt - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def render_one(src: Path, dets: list, out_path: Path):
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        return False
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not vw.isOpened():
        cap.release()
        return False
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        draw_boxes(frame, dets)
        vw.write(frame)
    cap.release()
    vw.release()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out-dir', default=str(OUT_DIR))
    ap.add_argument('--src-dir', default=str(SRC_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = [json.loads(l) for l in open(DETS)]
    rng = random.Random(args.seed)
    sample = rng.sample(records, min(args.n, len(records)))

    n_ok = n_fail = 0
    src_dir = Path(args.src_dir)
    for i, rec in enumerate(sample, 1):
        src = src_dir / rec['video']
        out = out_dir / (Path(rec['video']).stem + '_vis.mp4')
        if out.exists():
            n_ok += 1
            continue
        ok = render_one(src, rec['detections'], out)
        if ok:
            n_ok += 1
        else:
            n_fail += 1
            print(f'  FAIL {rec["video"]}', flush=True)
        if i % 25 == 0 or i == len(sample):
            print(f'  {i:4d}/{len(sample)}  ok={n_ok}  fail={n_fail}', flush=True)


if __name__ == '__main__':
    main()
