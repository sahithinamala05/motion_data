"""Generalization spot-check + keypoint export.

Samples N evenly-spaced videos across SRC_DIR (sorted alphabetically), runs
the pose model on the middle frame of each, and writes per-video outputs:
  - annotated PNG: <out>/<stem>.png  (boxes + keypoints overlaid)
  - JSON:          <out>/json/<stem>.json
      {
        "video": "<name>.mp4",
        "frame_idx": <int>,
        "width": <int>, "height": <int>,
        "detections": {
          "shoe":    [{"box_conf": ..., "bbox_xyxy": [...],
                       "keypoints": [{"id": 1, "x": ..., "y": ..., "conf": ...}, ...7]}],
          "scanner": [...same shape, only id 1..4 will be confident...]
        }
      }
Only "shoe" and "scanner" classes are written to JSON (per request).
"""
import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

WEIGHTS = Path('/home/ubuntu/us-west-3-fs/sahithi/static_obj_training/model/best.pt')
SRC_DIR = Path('/home/ubuntu/us-west-3-fs/somrita/livedealer/roundcut/batch_01')
OUT_DIR = Path('/home/ubuntu/us-west-3-fs/sahithi/inf_objects')

KEEP_CLASSES = {'shoe', 'scanner'}
COLORS = {
    'discard_tray': (0, 215, 255),
    'shoe':         (0, 255,   0),
    'spare_shoe':   (255, 128,  0),
    'scanner':      (0,   0, 255),
}
KP_VIS_THR = 0.5


def middle_frame(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None, None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        cap.release()
        return None, None
    mid = n // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
    ok, frame = cap.read()
    cap.release()
    return (frame, mid) if ok else (None, None)


def annotate(frame, r, names):
    if r.boxes is None or len(r.boxes) == 0:
        return frame
    boxes = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    clss = r.boxes.cls.cpu().numpy().astype(int)
    kxy = r.keypoints.xy.cpu().numpy() if r.keypoints is not None else None
    kconf = r.keypoints.conf
    kconf = kconf.cpu().numpy() if kconf is not None else None

    for i, (box, c, k_id) in enumerate(zip(boxes, confs, clss)):
        cname = names[int(k_id)]
        color = COLORS.get(cname, (255, 255, 255))
        if kxy is not None:
            for j, (x, y) in enumerate(kxy[i]):
                if kconf is not None and kconf[i, j] < KP_VIS_THR:
                    continue
                if x == 0 and y == 0:
                    continue
                cx, cy = int(x), int(y)
                cv2.circle(frame, (cx, cy), 5, color, -1)
                cv2.circle(frame, (cx, cy), 6, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, str(j + 1), (cx + 5, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        x1, y1, _, _ = map(int, box)
        label = f'{cname} {float(c):.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ytxt = max(th + 4, y1)
        cv2.rectangle(frame, (x1, ytxt - th - 4), (x1 + tw + 4, ytxt), color, -1)
        cv2.putText(frame, label, (x1 + 2, ytxt - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def result_to_json(r, names, frame_idx: int):
    h, w = r.orig_shape
    out = {'frame_idx': int(frame_idx), 'width': int(w), 'height': int(h),
           'detections': {c: [] for c in KEEP_CLASSES}}
    if r.boxes is None or len(r.boxes) == 0:
        return out
    boxes = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    clss = r.boxes.cls.cpu().numpy().astype(int)
    kxy = r.keypoints.xy.cpu().numpy() if r.keypoints is not None else None
    kconf = r.keypoints.conf
    kconf = kconf.cpu().numpy() if kconf is not None else None

    for i, (box, c, k_id) in enumerate(zip(boxes, confs, clss)):
        cname = names[int(k_id)]
        if cname not in KEEP_CLASSES:
            continue
        kpts = []
        if kxy is not None:
            for j, (x, y) in enumerate(kxy[i]):
                kpts.append({
                    'id': j + 1,
                    'x': float(x),
                    'y': float(y),
                    'conf': float(kconf[i, j]) if kconf is not None else None,
                })
        out['detections'][cname].append({
            'box_conf': float(c),
            'bbox_xyxy': [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
            'keypoints': kpts,
        })
    return out


def pick_videos(src: Path, n: int):
    vids = sorted(src.glob('*.mp4'))
    if len(vids) <= n:
        return vids
    # evenly-spaced indices
    step = (len(vids) - 1) / (n - 1)
    idxs = [round(i * step) for i in range(n)]
    return [vids[i] for i in idxs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default=str(WEIGHTS))
    ap.add_argument('--src-dir', default=str(SRC_DIR))
    ap.add_argument('--out-dir', default=str(OUT_DIR))
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--imgsz', type=int, default=1280)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--device', default='0')
    args = ap.parse_args()

    out = Path(args.out_dir)
    (out / 'json').mkdir(parents=True, exist_ok=True)

    vids = pick_videos(Path(args.src_dir), args.n)
    print(f'Running pose on {len(vids)} videos (evenly spaced) -> {out}')

    model = YOLO(args.weights)
    names = model.names

    for i, v in enumerate(vids, 1):
        frame, fi = middle_frame(v)
        if frame is None:
            print(f'[{i}/{len(vids)}] {v.name}  FAIL (no frame)')
            continue
        r = model(frame, imgsz=args.imgsz, conf=args.conf,
                  device=args.device, verbose=False)[0]

        annotated = annotate(frame.copy(), r, names)
        cv2.imwrite(str(out / (v.stem + '.png')), annotated)

        rec = result_to_json(r, names, fi)
        rec['video'] = v.name
        (out / 'json' / (v.stem + '.json')).write_text(json.dumps(rec, indent=2))

        n_shoe = len(rec['detections']['shoe'])
        n_scan = len(rec['detections']['scanner'])
        print(f'[{i}/{len(vids)}] {v.name}  shoe={n_shoe}  scanner={n_scan}')


if __name__ == '__main__':
    main()
