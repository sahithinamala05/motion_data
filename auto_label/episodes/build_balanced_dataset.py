"""Build a balanced training set: equal human-annotated + auto-cluster entries.

Sources:
  - LS export -> N human-corrected entries (rotated corners)
  - split300 cluster JSON -> N auto-cluster entries (axis-aligned)
Sample so both contribute the same count = min(|human|, |auto|).

Also stages all referenced PNGs into a single dir via symlinks so
build_expanded_dataset.py + train_yolo_obb.py find them.
"""

import argparse
import json
import math
import os
import random
from pathlib import Path

CLASS_MAPPING = {
    "Dealer": 0, "P1": 1, "P2": 2, "P3": 3,
    "P4": 4, "P5": 5, "P6": 6, "P7": 7,
}


def rotated_corners(x_pct, y_pct, w_pct, h_pct, rot_deg, img_w, img_h):
    px = x_pct * img_w / 100.0
    py = y_pct * img_h / 100.0
    pw = w_pct * img_w / 100.0
    ph = h_pct * img_h / 100.0
    base = [(0.0, 0.0), (pw, 0.0), (pw, ph), (0.0, ph)]
    t = math.radians(rot_deg)
    c, s = math.cos(t), math.sin(t)
    out = []
    for dx, dy in base:
        rx = dx * c - dy * s
        ry = dx * s + dy * c
        out.append([px + rx, py + ry])
    return out


def pick_annotation(task):
    anns = [a for a in task.get("annotations", [])
            if not a.get("was_cancelled") and a.get("result")]
    if not anns:
        return None
    anns.sort(key=lambda a: a.get("updated_at") or a.get("created_at") or "")
    return anns[-1]


def task_to_entry(task, img_w, img_h):
    png = task["data"]["image"].rsplit("/", 1)[-1]
    ann = pick_annotation(task)
    if ann is None:
        return None, "no_annotation"
    boxes = []
    for r in ann["result"]:
        if r.get("type") != "rectanglelabels":
            continue
        v = r["value"]
        labels = v.get("rectanglelabels", [])
        if not labels:
            continue
        cls = labels[0]
        if cls not in CLASS_MAPPING:
            continue
        ow = r.get("original_width", img_w)
        oh = r.get("original_height", img_h)
        corners = rotated_corners(v["x"], v["y"], v["width"], v["height"],
                                  v.get("rotation", 0.0) or 0.0, ow, oh)
        sx = img_w / ow
        sy = img_h / oh
        corners = [[round(x * sx, 3), round(y * sy, 3)] for x, y in corners]
        boxes.append({"class": cls, "class_id": CLASS_MAPPING[cls], "corners": corners})
    if not boxes:
        return None, "no_boxes"
    return {"image": png, "boxes": boxes}, "ok"


def load_ls_entries(ls_path, img_w, img_h):
    tasks = json.load(open(ls_path))
    out = []
    for t in tasks:
        e, status = task_to_entry(t, img_w, img_h)
        if e is not None:
            out.append(e)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ls_json",
                   default="/home/ubuntu/sahithi/motion-data-process/manual_label/data/all_clus_anno.json")
    p.add_argument("--auto_json",
                   default="/home/ubuntu/us-west-3-fs/sahithi/clus_anno/split300_clustered_annotations.json")
    p.add_argument("--human_src_dir",
                   default="/home/ubuntu/us-west-3-fs/sahithi/clus_anno/to_annotate")
    p.add_argument("--auto_src_dir",
                   default="/home/ubuntu/us-west-3-fs/sahithi/clus_anno/split300")
    p.add_argument("--out_json",
                   default="/home/ubuntu/us-west-3-fs/sahithi/clus_anno/all_clus_anno_balanced.json")
    p.add_argument("--out_images_dir",
                   default="/home/ubuntu/us-west-3-fs/sahithi/clus_anno/to_annotate_balanced")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    auto = json.load(open(args.auto_json))
    img_w = auto["image_size"]["width"]
    img_h = auto["image_size"]["height"]
    classes = auto["classes"]

    human = load_ls_entries(args.ls_json, img_w, img_h)
    auto_entries = list(auto["annotations"])

    n = min(len(human), len(auto_entries))
    rng = random.Random(args.seed)
    rng.shuffle(human)
    rng.shuffle(auto_entries)
    human = human[:n]
    auto_entries = auto_entries[:n]
    print(f"human: {len(human)}  auto: {len(auto_entries)}  -> balanced N = {n}")

    merged = human + auto_entries

    # Stage symlinks
    out_imgs = Path(args.out_images_dir)
    out_imgs.mkdir(parents=True, exist_ok=True)
    n_linked = 0
    for e in human:
        src = Path(args.human_src_dir) / e["image"]
        dst = out_imgs / e["image"]
        if not dst.exists():
            os.symlink(src, dst)
            n_linked += 1
    for e in auto_entries:
        src = Path(args.auto_src_dir) / e["image"]
        dst = out_imgs / e["image"]
        if not dst.exists():
            os.symlink(src, dst)
            n_linked += 1
    print(f"symlinks staged: {n_linked} new (total in dir: {len(list(out_imgs.iterdir()))})")

    json.dump(
        {"image_size": auto["image_size"], "classes": classes, "annotations": merged},
        open(args.out_json, "w"), indent=2,
    )
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
