"""Run YOLO on the FIRST frame of every roundcut video. For each video where
all 4 classes (discard_tray, shoe, spare_shoe, scanner) are detected, write
one JSON file per video to the output directory.

Per-video JSON schema:
{
  "video": "<filename>",
  "frame_idx": 0,
  "width": 1280, "height": 720,
  "detections": [
     {"class": "shoe", "conf": 0.98, "bbox_xyxy": [x1, y1, x2, y2]},
     ...
  ]
}
For multi-detection classes (e.g. two discard_tray boxes), only the highest-conf
box per class is kept so each JSON has exactly 4 detections.
"""
import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

WEIGHTS = Path('/home/ubuntu/us-west-3-fs/sahithi/shoe_scanner/model/best.pt')
LIST_FILE = Path('/home/ubuntu/us-west-3-fs/live_dealer_blackjack/roundcut/good_quality_rounds_video_list.txt')
OUT_DIR = Path('/home/ubuntu/us-west-3-fs/sahithi/shoe_scanner/json')
CLASSES = ('discard_tray', 'shoe', 'spare_shoe', 'scanner')
CONF_THR = 0.25
BATCH = 32
IMGSZ = 1280


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default=str(WEIGHTS))
    ap.add_argument('--list', default=str(LIST_FILE))
    ap.add_argument('--out-dir', default=str(OUT_DIR))
    ap.add_argument('--conf', type=float, default=CONF_THR)
    ap.add_argument('--batch', type=int, default=BATCH)
    ap.add_argument('--imgsz', type=int, default=IMGSZ)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--device', default='0')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vids = [Path(p.strip()) for p in Path(args.list).read_text().splitlines() if p.strip()]
    if args.limit:
        vids = vids[:args.limit]
    print(f'Videos to process: {len(vids)}')

    model = YOLO(args.weights)
    names = model.names

    n_saved = n_missing_class = n_decode_fail = 0
    t0 = time.time()

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for batch_start in range(0, len(vids), args.batch):
            batch = vids[batch_start:batch_start + args.batch]
            frames, kept = [], []
            for v in batch:
                png = td / (v.stem + '.png')
                cap = cv2.VideoCapture(str(v))
                if not cap.isOpened():
                    n_decode_fail += 1
                    continue
                ok, frame = cap.read()  # 1st frame
                cap.release()
                if not ok or frame is None:
                    n_decode_fail += 1
                    continue
                cv2.imwrite(str(png), frame)
                frames.append(str(png))
                kept.append(v)

            if not frames:
                continue

            results = model(frames, imgsz=args.imgsz, conf=args.conf,
                            device=args.device, verbose=False)
            for v, r in zip(kept, results):
                h, w = r.orig_shape
                best_per_class = {}
                if r.boxes is not None and len(r.boxes) > 0:
                    xyxy = r.boxes.xyxy.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                    clss = r.boxes.cls.cpu().numpy().astype(int)
                    for box, c, cls in zip(xyxy, confs, clss):
                        cname = names[int(cls)]
                        if cname not in CLASSES:
                            continue
                        prev = best_per_class.get(cname)
                        if prev is None or float(c) > prev['conf']:
                            best_per_class[cname] = {
                                'class': cname,
                                'conf': float(c),
                                'bbox_xyxy': [float(box[0]), float(box[1]),
                                              float(box[2]), float(box[3])],
                            }
                if not all(c in best_per_class for c in CLASSES):
                    n_missing_class += 1
                    continue
                dets = [best_per_class[c] for c in CLASSES]
                rec = {
                    'video': v.name,
                    'frame_idx': 0,
                    'width': int(w),
                    'height': int(h),
                    'detections': dets,
                }
                (out_dir / (v.stem + '.json')).write_text(json.dumps(rec, indent=2))
                n_saved += 1

            for p in frames:
                try: os.remove(p)
                except OSError: pass

            elapsed = time.time() - t0
            done = batch_start + len(batch)
            print(f'  {done:5d}/{len(vids)}  saved={n_saved}  '
                  f'missing_class={n_missing_class}  decode_fail={n_decode_fail}  '
                  f'elapsed={elapsed:.0f}s', flush=True)

    print(f'Done. saved={n_saved}  missing_class={n_missing_class}  '
          f'decode_fail={n_decode_fail}  out={out_dir}')


if __name__ == '__main__':
    main()
