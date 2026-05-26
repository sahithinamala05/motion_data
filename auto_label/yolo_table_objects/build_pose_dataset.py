"""Convert Label Studio keypoint annotations to YOLO POSE format.

Layout per label line (one line per object instance):
  cls_id  cx cy w h  kp1_x kp1_y kp1_v ... kp7_x kp7_y kp7_v

All coords normalized 0-1. Visibility flag v:
  2 = labeled & visible, 0 = missing/invisible (kp coords set to 0)

Slot mapping by C-index: "C1"→slot 0 ... "C7"→slot 6.
- discard_tray, shoe: fill 7 slots from labels (drop any C# > 7).
- spare_shoe, scanner: fill slots 0..3 from C1..C4; slots 4..6 set to (0,0,0).

Bbox is the tight box around the visible keypoints, padded by PAD_PCT of the
frame dimension (small pad so YOLO has non-trivial area to train on).
"""
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

ANNO_JSON = '/home/ubuntu/sahithi/motion-data-process/manual_label/data/all_clus_anno.json'
IMG_SRC   = Path('/home/ubuntu/us-west-3-fs/sahithi/clus_anno/to_annotate')
DATASET   = Path('/home/ubuntu/sahithi/motion-data-process/auto_label/yolo_table_objects/dataset_pose')

CLASSES   = ['discard_tray', 'shoe', 'spare_shoe', 'scanner']
CLS_ID    = {c: i for i, c in enumerate(CLASSES)}
N_KPTS    = 7
PAD_PCT   = 0.01
MIN_VIS   = 2          # need at least 2 visible keypoints to make a usable instance
VAL_FRAC  = 0.20
SEED      = 42


def parse_c_index(label: str) -> int:
    """C1→0, C7→6. Returns -1 if not a valid Cn label or out of range."""
    if not label.startswith('C'):
        return -1
    try:
        idx = int(label[1:]) - 1
    except ValueError:
        return -1
    return idx if 0 <= idx < N_KPTS else -1


def task_to_yolo(entry):
    img_bn = os.path.basename(entry['data']['image'])
    iw = ih = None
    # cls -> [(slot, x_pct, y_pct)]
    by_cls = defaultdict(list)
    for ann in entry.get('annotations', []):
        for r in ann.get('result', []):
            if r.get('type') != 'keypointlabels':
                continue
            fn = r.get('from_name')
            if fn not in CLS_ID:
                continue
            v = r['value']
            labs = v.get('keypointlabels') or []
            if not labs:
                continue
            slot = parse_c_index(labs[0])
            if slot < 0:
                continue
            by_cls[fn].append((slot, v['x'], v['y']))
            iw = iw or r.get('original_width')
            ih = ih or r.get('original_height')

    lines = []
    for cname, pts in by_cls.items():
        # de-dup: keep latest entry per slot
        slot_to_xy = {}
        for slot, x, y in pts:
            slot_to_xy[slot] = (x, y)
        visible = list(slot_to_xy.values())
        if len(visible) < MIN_VIS:
            continue
        xs = [p[0] for p in visible]
        ys = [p[1] for p in visible]
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

        kp_tokens = []
        for slot in range(N_KPTS):
            if slot in slot_to_xy:
                x, y = slot_to_xy[slot]
                kp_tokens.append(f'{x/100.0:.6f} {y/100.0:.6f} 2')
            else:
                kp_tokens.append('0.000000 0.000000 0')
        cls_id = CLS_ID[cname]
        lines.append(f'{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} ' + ' '.join(kp_tokens))
    return img_bn, lines


def main():
    if DATASET.exists():
        shutil.rmtree(DATASET)
    for split in ('train', 'val'):
        (DATASET / 'images' / split).mkdir(parents=True)
        (DATASET / 'labels' / split).mkdir(parents=True)

    with open(ANNO_JSON) as f:
        tasks = json.load(f)
    rng = random.Random(SEED)
    rng.shuffle(tasks)
    n_val = int(len(tasks) * VAL_FRAC)
    splits = {'val': tasks[:n_val], 'train': tasks[n_val:]}

    inst_per_class = defaultdict(int)
    vis_kpts_per_class = defaultdict(int)
    n_skipped_no_img = n_skipped_no_lab = 0
    for split, items in splits.items():
        for e in items:
            img_bn, lines = task_to_yolo(e)
            src = IMG_SRC / img_bn
            if not src.exists():
                n_skipped_no_img += 1
                continue
            if not lines:
                n_skipped_no_lab += 1
                continue
            dst_img = DATASET / 'images' / split / img_bn
            dst_lbl = DATASET / 'labels' / split / (Path(img_bn).stem + '.txt')
            if not dst_img.exists():
                os.symlink(src, dst_img)
            dst_lbl.write_text('\n'.join(lines) + '\n')
            for ln in lines:
                toks = ln.split()
                cls = CLASSES[int(toks[0])]
                inst_per_class[cls] += 1
                # kpt tokens start after 1 cls + 4 box = 5
                kpts = toks[5:]
                for i in range(0, len(kpts), 3):
                    if int(kpts[i + 2]) == 2:
                        vis_kpts_per_class[cls] += 1

    print('train images:', len(list((DATASET / 'images' / 'train').iterdir())))
    print('val   images:', len(list((DATASET / 'images' / 'val').iterdir())))
    print('skipped (no image):', n_skipped_no_img, ' (no labels):', n_skipped_no_lab)
    print('instances per class:', dict(inst_per_class))
    print('visible-kpts per class:', dict(vis_kpts_per_class))


if __name__ == '__main__':
    main()
