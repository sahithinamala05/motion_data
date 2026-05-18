"""Run YOLO-OBB inference on a directory of images and save annotated frames.

Defaults are tuned for the human_v2 seat-detection model.
"""

import argparse
import os
from pathlib import Path

from ultralytics import YOLO

DEFAULT_MODEL = "/home/ubuntu/us-west-3-fs/sahithi/yolo_obb_runs/human_v2/weights/best.pt"
DEFAULT_SRC = "/home/ubuntu/us-west-3-fs/sahithi/clus_anno/crowded"
DEFAULT_OUT = "/home/ubuntu/us-west-3-fs/sahithi/clus_anno/crowded_inf"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--src", default=DEFAULT_SRC)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--device", default="0")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    images = sorted(
        str(p) for p in Path(args.src).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    print(f"Found {len(images)} images in {args.src}")

    model = YOLO(args.model)
    n_total = 0
    for i, img_path in enumerate(images, 1):
        results = model.predict(
            source=img_path,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )
        r = results[0]
        n_det = 0 if r.obb is None else len(r.obb)
        n_total += n_det
        save_path = out / Path(img_path).name
        r.save(filename=str(save_path))
        if i % 25 == 0 or i == len(images):
            print(f"  [{i}/{len(images)}] last={n_det} dets, total={n_total}", flush=True)

    print(f"\nSaved {len(images)} annotated images to {out} (total detections: {n_total})")


if __name__ == "__main__":
    main()
