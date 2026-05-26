"""YOLO inference on the middle frame of the first N videos in batch_01.
Writes one annotated PNG per video to inf_objects/.
"""
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

WEIGHTS = Path('/home/ubuntu/sahithi/motion-data-process/auto_label/yolo_table_objects/runs/table_objects/weights/best.pt')
SRC_DIR = Path('/home/ubuntu/us-west-3-fs/somrita/livedealer/roundcut/batch_01')
OUT_DIR = Path('/home/ubuntu/us-west-3-fs/sahithi/inf_objects')

COLORS = {
    'discard_tray': (0, 215, 255),   # gold
    'shoe':         (0, 255,   0),   # green
    'spare_shoe':   (255, 128,  0),  # blue-cyan
    'scanner':      (0,   0, 255),   # red
}


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
    for box, c, k in zip(r.boxes.xyxy.cpu().numpy(),
                          r.boxes.conf.cpu().numpy(),
                          r.boxes.cls.cpu().numpy().astype(int)):
        x1, y1, x2, y2 = box
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cname = names[int(k)]
        color = COLORS.get(cname, (255, 255, 255))
        cv2.circle(frame, (cx, cy), 6, color, -1)
        cv2.circle(frame, (cx, cy), 7, (0, 0, 0), 1, cv2.LINE_AA)
        label = f'{cname} {float(c):.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        tx, ty = cx + 8, cy - 8
        cv2.rectangle(frame, (tx - 2, ty - th - 2), (tx + tw + 2, ty + 2), color, -1)
        cv2.putText(frame, label, (tx, ty),
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
    print(f'Running on {len(vids)} videos -> {out}')

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
