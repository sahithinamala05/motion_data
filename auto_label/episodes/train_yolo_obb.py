"""
Build a YOLO-OBB dataset from clustered-annotations JSON, following the layout
documented at https://docs.ultralytics.com/datasets/obb/.

Workflow mirrors the official DOTA -> YOLO-OBB pipeline:

    <out>/
      images/{train,val}/<png>
      labels/{train,val}_original/<txt>   # DOTA-style: 8 abs coords + class + diff
      labels/{train,val}/<txt>            # YOLO-OBB: class_idx + 8 normalized coords

The standard ultralytics helper `convert_dota_to_yolo_obb` only knows the 18
default DOTA classes, so we replicate its conversion with our seat class map.

Install if needed: pip install ultralytics
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path

CLASS_MAPPING = {
    "Dealer": 0, "P1": 1, "P2": 2, "P3": 3,
    "P4": 4, "P5": 5, "P6": 6, "P7": 7,
}


def write_dota_label(path, boxes):
    """DOTA format: one line per box -> x1 y1 x2 y2 x3 y3 x4 y4 class_name diff

    Accepts boxes in either form:
      - {"corners": [[x,y]*4], ...}  (rotated, preferred)
      - {"x1","y1","x2","y2", ...}    (axis-aligned legacy)
    """
    with open(path, "w") as f:
        for b in boxes:
            if "corners" in b:
                flat = [v for pt in b["corners"] for v in pt]
            else:
                x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
                flat = [x1, y1, x2, y1, x2, y2, x1, y2]
            f.write(" ".join(f"{c:.3f}" for c in flat) + f" {b['class']} 0\n")


def dota_to_yolo_obb(orig_label_path, save_path, image_w, image_h, class_mapping):
    """Same logic as ultralytics.data.converter.convert_dota_to_yolo_obb's
    inner convert_label, but with a caller-supplied class mapping."""
    with open(orig_label_path) as fin, open(save_path, "w") as fout:
        for line in fin:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            class_name = parts[8]
            class_idx = class_mapping[class_name]
            coords = [float(p) for p in parts[:8]]
            normalized = [
                coords[i] / image_w if i % 2 == 0 else coords[i] / image_h
                for i in range(8)
            ]
            fout.write(
                f"{class_idx} " + " ".join(f"{c:.6g}" for c in normalized) + "\n"
            )


def build_dataset(ann_json_path, images_src_dir, out_dir, val_frac, seed):
    with open(ann_json_path) as f:
        data = json.load(f)
    img_w = data["image_size"]["width"]
    img_h = data["image_size"]["height"]
    classes = data["classes"]
    items = data["annotations"]

    # If entries carry an explicit "split" field, honor it; otherwise fall back
    # to random shuffled split.
    if any("split" in it for it in items):
        splits = {"train": [], "val": []}
        for it in items:
            splits.setdefault(it.get("split", "train"), []).append(it)
        print(f"Pre-split: train={len(splits['train'])} val={len(splits['val'])}")
    else:
        rng = random.Random(seed)
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_frac))
        splits = {"val": items[:n_val], "train": items[n_val:]}
        print(f"Random split: train={len(splits['train'])} val={len(splits['val'])}")

    out = Path(out_dir)
    for split, entries in splits.items():
        img_dir = out / "images" / split
        orig_lbl_dir = out / "labels" / f"{split}_original"
        yolo_lbl_dir = out / "labels" / split
        for d in (img_dir, orig_lbl_dir, yolo_lbl_dir):
            d.mkdir(parents=True, exist_ok=True)

        for entry in entries:
            png = entry["image"]
            stem = png.rsplit(".", 1)[0]

            src_img = Path(images_src_dir) / png
            dst_img = img_dir / png
            if not dst_img.exists():
                os.symlink(src_img, dst_img)

            orig_label = orig_lbl_dir / f"{stem}.txt"
            yolo_label = yolo_lbl_dir / f"{stem}.txt"
            write_dota_label(orig_label, entry["boxes"])
            dota_to_yolo_obb(orig_label, yolo_label, img_w, img_h, CLASS_MAPPING)

    yaml_path = out / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {out.resolve()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names:\n")
        for i, name in enumerate(classes):
            f.write(f"  {i}: {name}\n")
    print(f"Wrote dataset.yaml -> {yaml_path}")
    return str(yaml_path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ann_json",
                   default="/home/ubuntu/us-west-3-fs/sahithi/clus_anno/to_annotate_clustered_annotations.json")
    p.add_argument("--images_dir",
                   default="/home/ubuntu/us-west-3-fs/sahithi/clus_anno/to_annotate")
    p.add_argument("--out_dir",
                   default="/home/ubuntu/us-west-3-fs/sahithi/yolo_obb_seat_dataset")
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", default="yolo26n-obb.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="0")
    p.add_argument("--project",
                   default="/home/ubuntu/us-west-3-fs/sahithi/yolo_obb_runs")
    p.add_argument("--name", default="seat_clustering_v1")
    p.add_argument("--rebuild", action="store_true",
                   help="delete existing out_dir and rebuild from scratch")
    p.add_argument("--no_train", action="store_true",
                   help="only build the dataset, skip training")
    args = p.parse_args()

    if args.rebuild and os.path.exists(args.out_dir):
        shutil.rmtree(args.out_dir)

    yaml_path = build_dataset(
        args.ann_json, args.images_dir, args.out_dir, args.val_frac, args.seed
    )

    if args.no_train:
        return

    from ultralytics import YOLO
    model = YOLO(args.model)
    model.train(
        data=yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
