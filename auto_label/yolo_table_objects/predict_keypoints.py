"""Predict scanner + shoe keypoints from a single frame using the trained
YOLOv8 pose model.

Usage:
    from predict_keypoints import load_model, predict_keypoints
    model = load_model()                       # once per process
    kpts  = predict_keypoints(model, frame)    # frame is a BGR numpy array

Returns:
    {
        "scanner": {1: (x, y, conf), 2: ..., 3: ..., 4: ...} | None,
        "shoe":    {1: (x, y, conf), ..., 7: ...}             | None,
    }

If a class is not detected, its value is None. Keypoints with conf below
KP_CONF_THR are dropped, so scanner naturally has only ids 1..4.

Notes:
- The model file path below assumes the bundle at
  /home/ubuntu/us-west-3-fs/sahithi/static_obj_training/. Override via the
  weights kwarg if you copy the .pt elsewhere.
- For multiple frames, batch them: model([frame1, frame2, ...]) is much
  faster than calling predict_keypoints in a loop.
"""
from pathlib import Path
from ultralytics import YOLO

DEFAULT_WEIGHTS = '/home/ubuntu/us-west-3-fs/sahithi/static_obj_training/model/best.pt'
KP_CONF_THR     = 0.5     # drop low-confidence keypoints (filters scanner slots 5..7)
KEEP_CLASSES    = ('scanner', 'shoe')


def load_model(weights: str = DEFAULT_WEIGHTS) -> YOLO:
    return YOLO(weights)


def predict_keypoints(model: YOLO, frame_bgr,
                      imgsz: int = 1280, conf: float = 0.25,
                      device: str = '0') -> dict:
    """Run the pose model on one BGR frame and return scanner + shoe keypoints.

    Args:
        model:     YOLO instance from load_model().
        frame_bgr: HxWx3 numpy array (cv2 BGR order).
        imgsz:     model input size (1280 matches training).
        conf:      box confidence threshold.
        device:    '0' for first GPU, 'cpu' to disable CUDA.

    Returns:
        dict {class_name: {kpt_id: (x, y, conf)}}. Missing classes -> None.
    """
    r = model(frame_bgr, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
    names = model.names
    out = {c: None for c in KEEP_CLASSES}

    if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None:
        return out

    boxes  = r.boxes.xyxy.cpu().numpy()
    bconf  = r.boxes.conf.cpu().numpy()
    clss   = r.boxes.cls.cpu().numpy().astype(int)
    kxy    = r.keypoints.xy.cpu().numpy()      # (N, K, 2) absolute pixel coords
    kconf  = r.keypoints.conf.cpu().numpy()    # (N, K)

    # group detections by class so we can keep the highest-conf instance per class
    per_class = {c: [] for c in KEEP_CLASSES}
    for i, k_id in enumerate(clss):
        cname = names[int(k_id)]
        if cname not in per_class:
            continue
        kpts = {}
        for j in range(kxy.shape[1]):
            if kconf[i, j] >= KP_CONF_THR:
                kpts[j + 1] = (float(kxy[i, j, 0]),
                               float(kxy[i, j, 1]),
                               float(kconf[i, j]))
        per_class[cname].append((float(bconf[i]), kpts))

    for cname, items in per_class.items():
        if not items:
            continue
        items.sort(key=lambda t: t[0], reverse=True)  # highest box conf wins
        out[cname] = items[0][1]
    return out


# ----- standalone demo -----
if __name__ == '__main__':
    import sys
    import cv2

    if len(sys.argv) < 2:
        print('usage: python predict_keypoints.py <video_or_image_path>')
        sys.exit(1)

    src = Path(sys.argv[1])
    if src.suffix.lower() in {'.mp4', '.mov', '.mkv', '.avi'}:
        cap = cv2.VideoCapture(str(src))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            sys.exit('could not read middle frame')
    else:
        frame = cv2.imread(str(src))
        if frame is None:
            sys.exit('could not read image')

    model = load_model()
    kpts  = predict_keypoints(model, frame)

    print('scanner:', kpts['scanner'])  # e.g. {1: (x,y,conf), 2: ..., 3: ..., 4: ...}
    print('shoe   :', kpts['shoe'])
    if kpts['shoe']:
        exit_kpts = {i: kpts['shoe'].get(i) for i in (1, 2, 3, 4)}
        print('shoe exit kpts (1..4):', exit_kpts)
