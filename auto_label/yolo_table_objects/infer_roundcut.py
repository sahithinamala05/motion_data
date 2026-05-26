"""Run trained YOLO table-objects model on the middle frame of every
roundcut video and write detections to a single JSONL.

Each line:
{
  "video": "<filename>",
  "frame_idx": <int>,
  "width": 1280, "height": 720,
  "detections": [
    {"class": "shoe", "conf": 0.98, "bbox_xyxy": [x1, y1, x2, y2]},
    ...
  ]
}

Bboxes are absolute pixel coords in the source frame.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

WEIGHTS = Path('/home/ubuntu/sahithi/motion-data-process/auto_label/yolo_table_objects/runs/table_objects/weights/best.pt')
VIDEO_DIR = Path('/home/ubuntu/us-west-3-fs/sahithi/../live_dealer_blackjack/roundcut/good_quality_rounds').resolve()
LIST_FILE = Path('/home/ubuntu/us-west-3-fs/live_dealer_blackjack/roundcut/good_quality_rounds_video_list.txt')
OUT_JSONL = Path('/home/ubuntu/sahithi/motion-data-process/auto_label/yolo_table_objects/roundcut_detections.jsonl')
FFMPEG = '/usr/bin/ffmpeg'
FFPROBE = '/usr/bin/ffprobe'
CONF_THR = 0.25
BATCH = 32


def video_frame_count(path: str) -> int:
    """Total frames via ffprobe (fast, no decode)."""
    try:
        out = subprocess.check_output(
            [FFPROBE, '-v', 'error', '-select_streams', 'v:0',
             '-count_packets', '-show_entries', 'stream=nb_read_packets',
             '-of', 'csv=p=0', path],
            stderr=subprocess.DEVNULL, text=True).strip()
        return int(out) if out else 0
    except Exception:
        return 0


def extract_middle_frame(path: str, out_png: str) -> bool:
    """Extract the middle frame using OpenCV (fast seek)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return False
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        cap.release()
        return False
    mid = n // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return False
    cv2.imwrite(out_png, frame)
    return True


def load_video_list(args) -> list[Path]:
    if args.list:
        paths = [Path(p.strip()) for p in Path(args.list).read_text().splitlines() if p.strip()]
    else:
        paths = sorted(Path(args.video_dir).glob('*.mp4'))
    if args.limit:
        paths = paths[:args.limit]
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default=str(WEIGHTS))
    ap.add_argument('--list', default=str(LIST_FILE))
    ap.add_argument('--video-dir', default=None)
    ap.add_argument('--out', default=str(OUT_JSONL))
    ap.add_argument('--conf', type=float, default=CONF_THR)
    ap.add_argument('--batch', type=int, default=BATCH)
    ap.add_argument('--imgsz', type=int, default=1280)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--device', default='0')
    args = ap.parse_args()

    vids = load_video_list(args)
    print(f'Videos to process: {len(vids)}')

    model = YOLO(args.weights)
    names = model.names

    out_f = open(args.out, 'w')
    n_ok, n_fail = 0, 0
    t0 = time.time()

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # process in batches: extract a batch of middle frames, run model
        for batch_start in range(0, len(vids), args.batch):
            batch = vids[batch_start:batch_start + args.batch]
            frames, kept, frame_idxs = [], [], []
            for v in batch:
                png = td / (v.stem + '.png')
                cap = cv2.VideoCapture(str(v))
                if not cap.isOpened():
                    n_fail += 1
                    continue
                n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if n <= 0:
                    cap.release()
                    n_fail += 1
                    continue
                mid = n // 2
                cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
                ok, frame = cap.read()
                cap.release()
                if not ok or frame is None:
                    n_fail += 1
                    continue
                cv2.imwrite(str(png), frame)
                frames.append(str(png))
                kept.append(v)
                frame_idxs.append(mid)

            if not frames:
                continue

            results = model(frames, imgsz=args.imgsz, conf=args.conf,
                            device=args.device, verbose=False)
            for v, fi, r in zip(kept, frame_idxs, results):
                h, w = r.orig_shape
                dets = []
                if r.boxes is not None and len(r.boxes) > 0:
                    xyxy = r.boxes.xyxy.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                    clss = r.boxes.cls.cpu().numpy().astype(int)
                    for box, c, cls in zip(xyxy, confs, clss):
                        dets.append({
                            'class': names[int(cls)],
                            'conf': float(c),
                            'bbox_xyxy': [float(box[0]), float(box[1]),
                                          float(box[2]), float(box[3])],
                        })
                out_f.write(json.dumps({
                    'video': v.name,
                    'frame_idx': int(fi),
                    'width': int(w),
                    'height': int(h),
                    'detections': dets,
                }) + '\n')
                n_ok += 1

            elapsed = time.time() - t0
            done = batch_start + len(batch)
            print(f'  {done:5d}/{len(vids)}  ok={n_ok}  fail={n_fail}  '
                  f'elapsed={elapsed:.0f}s  rate={done/max(elapsed,1e-3):.1f} vid/s',
                  flush=True)

            # remove pngs so /tmp doesn't fill up
            for p in frames:
                try: os.remove(p)
                except OSError: pass

    out_f.close()
    print(f'Done. ok={n_ok}  fail={n_fail}  out={args.out}')


if __name__ == '__main__':
    main()
