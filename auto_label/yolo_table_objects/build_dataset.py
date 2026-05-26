"""Convert Label Studio keypoint annotations to YOLO detection format.

Source : all_clus_anno.json  (Label Studio export, keypoint percentages 0-100)
Output : YOLO det dataset under DATASET_ROOT with 80/20 train/val split
Classes: 0=discard_tray  1=shoe  2=spare_shoe  3=scanner

Each instance's bbox is the min/max enclosing rectangle of its keypoints,
padded by PAD_PCT of the frame dimension. Instances with <MIN_POINTS
keypoints are dropped.
"""
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

ANNO_JSON = '/home/ubuntu/sahithi/motion-data-process/manual_label/data/all_clus_anno.json'
IMG_SRC   = Path('/home/ubuntu/us-west-3-fs/sahithi/clus_anno/to_annotate')
DATASET   = Path('/home/ubuntu/sahithi/motion-data-process/auto_label/yolo_table_objects/dataset')

CLASSES   = ['discard_tray', 'shoe', 'spare_shoe', 'scanner']
CLS_ID    = {c: i for i, c in enumerate(CLASSES)}
MIN_POINTS = 2     # need ≥2 points to define a box
PAD_PCT   = 0.01   # 1% of frame size, added on each side
VAL_FRAC  = 0.20
SEED      = 42


def load_tasks():
    with open(ANNO_JSON) as f:
        return json.load(f)


def task_to_yolo(entry):
    """Return (image_basename, [yolo_label_lines])."""
    img_bn = os.path.basename(entry['data']['image'])
    pts_by_obj = defaultdict(list)  # from_name -> list[(x_pct, y_pct)]
    iw = ih = None
    for ann in entry.get('annotations', []):
        for r in ann.get('result', []):
            if r.get('type') != 'keypointlabels':
                continue
            fn = r.get('from_name')
            if fn not in CLS_ID:
                continue
            v = r['value']
            pts_by_obj[fn].append((v['x'], v['y']))
            iw = iw or r.get('original_width')
            ih = ih or r.get('original_height')
    lines = []
    for fn, pts in pts_by_obj.items():
        if len(pts) < MIN_POINTS:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        # min/max in percent (0-100), then pad and convert to normalized 0-1
        x_min = max(0.0, min(xs) / 100.0 - PAD_PCT)
        x_max = min(1.0, max(xs) / 100.0 + PAD_PCT)
        y_min = max(0.0, min(ys) / 100.0 - PAD_PCT)
        y_max = min(1.0, max(ys) / 100.0 + PAD_PCT)
        w = x_max - x_min
        h = y_max - y_min
        if w <= 0 or h <= 0:
            continue
        cx = x_min + w / 2
        cy = y_min + h / 2
        lines.append(f'{CLS_ID[fn]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}')
    return img_bn, lines


def main():
    if DATASET.exists():
        shutil.rmtree(DATASET)
    for split in ('train', 'val'):
        (DATASET / 'images' / split).mkdir(parents=True)
        (DATASET / 'labels' / split).mkdir(parents=True)

    tasks = load_tasks()
    rng = random.Random(SEED)
    rng.shuffle(tasks)
    n_val = int(len(tasks) * VAL_FRAC)
    splits = {'val': tasks[:n_val], 'train': tasks[n_val:]}

    n_inst = defaultdict(int)
    n_skipped = 0
    for split, items in splits.items():
        for e in items:
            img_bn, lines = task_to_yolo(e)
            src = IMG_SRC / img_bn
            if not src.exists():
                n_skipped += 1
                continue
            if not lines:
                n_skipped += 1
                continue
            dst_img = DATASET / 'images' / split / img_bn
            dst_lbl = DATASET / 'labels' / split / (Path(img_bn).stem + '.txt')
            if not dst_img.exists():
                os.symlink(src, dst_img)
            dst_lbl.write_text('\n'.join(lines) + '\n')
            for ln in lines:
                n_inst[CLASSES[int(ln.split()[0])]] += 1

    print('train images:', len(list((DATASET / 'images' / 'train').iterdir())))
    print('val   images:', len(list((DATASET / 'images' / 'val').iterdir())))
    print('skipped (no image or no labels):', n_skipped)
    print('instances per class:', dict(n_inst))


if __name__ == '__main__':
    main()
