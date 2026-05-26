"""YOLO pose inference on the middle frame of the first N videos in batch_01.
Writes one annotated PNG per video to inf_objects/ with learned keypoints drawn.
"""
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

WEIGHTS = Path('/home/ubuntu/sahithi/motion-data-process/auto_label/yolo_table_objects/runs/table_objects_pose/weights/best.pt')
SRC_DIR = Path('/home/ubuntu/us-west-3-fs/somrita/livedealer/roundcut/batch_01')
OUT_DIR = Path('/home/ubuntu/us-west-3-fs/sahithi/inf_objects')

COLORS = {
    'discard_tray': (0, 215, 255),   # gold
    'shoe':         (0, 255,   0),   # green
    'spare_shoe':   (255, 128,  0),  # blue-cyan
    'scanner':      (0,   0, 255),   # red
}
KP_VIS_THR = 0.5


def middle_frame(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def annotate(frame, r, names):
    if r.boxes is None or len(r.boxes) == 0:
        return frame
    boxes = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    clss = r.boxes.cls.cpu().numpy().astype(int)
    if r.keypoints is None:
        return frame
    kxy = r.keypoints.xy.cpu().numpy()       # (N, K, 2)
    kconf = r.keypoints.conf
    kconf = kconf.cpu().numpy() if kconf is not None else None  # (N, K)

    for i, (box, c, k_id) in enumerate(zip(boxes, confs, clss)):
        cname = names[int(k_id)]
        color = COLORS.get(cname, (255, 255, 255))
        pts = kxy[i]
        for j, (x, y) in enumerate(pts):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default=str(WEIGHTS))
    ap.add_argument('--src-dir', default=str(SRC_DIR))
    ap.add_argument('--out-dir', default=str(OUT_DIR))
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--imgsz', type=int, default=1280)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--device', default='0')
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    vids = sorted(Path(args.src_dir).glob('*.mp4'))[:args.n]
    print(f'Running pose on {len(vids)} videos -> {out}')

    model = YOLO(args.weights)
    names = model.names

    for i, v in enumerate(vids, 1):
        frame = middle_frame(v)
        if frame is None:
            print(f'[{i}/{len(vids)}] {v.name}  FAIL (no frame)')
            continue
        r = model(frame, imgsz=args.imgsz, conf=args.conf,
                  device=args.device, verbose=False)[0]
        annotated = annotate(frame, r, names)
        dst = out / (v.stem + '.png')
        cv2.imwrite(str(dst), annotated)
        n_det = 0 if r.boxes is None else len(r.boxes)
        print(f'[{i}/{len(vids)}] {v.name}  detections={n_det}  -> {dst.name}')


if __name__ == '__main__':
    main()
