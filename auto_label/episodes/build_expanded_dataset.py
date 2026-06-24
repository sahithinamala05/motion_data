"""Expand the seat-OBB annotation dataset by duplicating split-flagged images.

Rule:
  - Images flagged as "split" get `--split_mult` copies; the rest get `--rest_mult`.
  - Flagging mode is `--match_mode round` (strip `_fNNN.png`, match by round name)
    or `filename` (require the exact PNG name to be in the split frames dir).

Duplicates are written as renamed copies (e.g. `<stem>_r1.png`) so train_yolo_obb
can symlink each one independently. The train/val split is done on the *unique*
image set so duplicates never leak between sides; val keeps originals only.
"""

import argparse
import json
import os
import random
import re
import shutil
from collections import Counter
from pathlib import Path


def round_from_image_name(image_name: str) -> str:
    """Strip the `_fNNN.png` suffix to get the round name."""
    return re.sub(r"_f\d+\.png$", "", image_name)


def load_split_keys(split_frames_dir: str, match_mode: str) -> set:
    """Return the set of keys (rounds or filenames) that identify a "split" image."""
    sd = Path(split_frames_dir)
    if not sd.exists():
        raise SystemExit(f"Missing {sd}")
    keys = set()
    for f in sd.iterdir():
        if match_mode == "round":
            m = re.match(r"(.+?)_f\d+\.png$", f.name)
            if m:
                keys.add(m.group(1))
        elif match_mode == "filename":
            if f.name.endswith(".png"):
                keys.add(f.name)
        else:
            raise SystemExit(f"Bad match_mode: {match_mode}")
    return keys


def is_split(entry: dict, split_keys: set, match_mode: str) -> bool:
    if match_mode == "round":
        return round_from_image_name(entry["image"]) in split_keys
    return entry["image"] in split_keys


def multiplier_for(entry: dict, split_keys: set, match_mode: str,
                   split_mult: int, rest_mult: int) -> int:
    return split_mult if is_split(entry, split_keys, match_mode) else rest_mult


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--in_json",
        default="/home/ubuntu/us-west-3-fs/sahithi/clus_anno/all_clus_anno_human_annotations.json",
        help="264-entry human-ground-truth annotations (no auto prelabels).",
    )
    p.add_argument(
        "--split_frames_dir",
        default="/home/ubuntu/us-west-3-fs/sahithi/clus_anno/split",
        help="PNG dir whose <round>_fNNN.png filenames identify split rounds.",
    )
    p.add_argument(
        "--src_images_dir",
        default="/home/ubuntu/us-west-3-fs/sahithi/clus_anno/to_annotate",
    )
    p.add_argument(
        "--out_json",
        default="/home/ubuntu/us-west-3-fs/sahithi/clus_anno/all_clus_anno_expanded.json",
    )
    p.add_argument(
        "--out_images_dir",
        default="/home/ubuntu/us-west-3-fs/sahithi/clus_anno/to_annotate_expanded",
    )
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split_mult", type=int, default=5,
                   help="copies of each split-flagged image (train only).")
    p.add_argument("--rest_mult", type=int, default=1,
                   help="copies of each non-split image (train only).")
    p.add_argument("--match_mode", choices=("round", "filename"), default="round",
                   help="'round' strips _fNNN.png and matches round names; "
                        "'filename' requires the exact PNG to live in split_frames_dir.")
    args = p.parse_args()

    src = json.load(open(args.in_json))
    split_keys = load_split_keys(args.split_frames_dir, args.match_mode)
    print(f"Loaded {len(src['annotations'])} annotations; "
          f"{len(split_keys)} split keys (mode={args.match_mode})")
    print(f"Multipliers: split×{args.split_mult}  rest×{args.rest_mult}")

    # Bucket by whether the image is flagged as a split, then stratify.
    buckets = {True: [], False: []}
    for e in src["annotations"]:
        buckets[is_split(e, split_keys, args.match_mode)].append(e)

    stats = {k: len(v) for k, v in buckets.items()}
    print(f"Unique images by split-match: {{in_split: {stats[True]}, not_split: {stats[False]}}}")

    rng = random.Random(args.seed)
    val_set, train_set = [], []
    for key, items in buckets.items():
        rng.shuffle(items)
        n_val = max(1, int(round(len(items) * args.val_frac))) if items else 0
        val_set.extend(items[:n_val])
        train_set.extend(items[n_val:])
    print(f"Unique split: train={len(train_set)} val={len(val_set)}")
    for key, items in buckets.items():
        nv = max(1, int(round(len(items) * args.val_frac))) if items else 0
        label = "in_split" if key else "not_split"
        print(f"  {label}: total={len(items)} val={nv} train={len(items)-nv}")

    out_dir = Path(args.out_images_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src_dir = Path(args.src_images_dir)
    out_anns = []

    # Val: originals only (no duplication)
    for e in val_set:
        png = e["image"]
        link = out_dir / png
        if not link.exists():
            os.symlink(src_dir / png, link)
        out_e = dict(e)
        out_e["split"] = "val"
        out_anns.append(out_e)

    # Train: split-flagged images appear split_mult times, others rest_mult times.
    for e in train_set:
        m = multiplier_for(e, split_keys, args.match_mode,
                           args.split_mult, args.rest_mult)
        stem, ext = e["image"].rsplit(".", 1)
        for k in range(m):
            new_name = e["image"] if k == 0 else f"{stem}_r{k}.{ext}"
            link = out_dir / new_name
            if not link.exists():
                os.symlink(src_dir / e["image"], link)
            out_e = dict(e)
            out_e["image"] = new_name
            out_e["split"] = "train"
            out_anns.append(out_e)

    out = {
        "image_size": src["image_size"],
        "classes": src["classes"],
        "annotations": out_anns,
    }
    json.dump(out, open(args.out_json, "w"), indent=2)
    expanded_train = sum(
        multiplier_for(e, split_keys, args.match_mode, args.split_mult, args.rest_mult)
        for e in train_set
    )
    print(f"\nWrote expanded dataset: {len(out_anns)} entries -> {args.out_json}")
    print(f"  unique val: {len(val_set)}")
    print(f"  expanded train: {expanded_train}  (vs {len(train_set)} unique)")
    print(f"Image symlinks dir: {out_dir}")


if __name__ == "__main__":
    main()
