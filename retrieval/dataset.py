"""
Dataset preparation for frame-level DINOv3 retrieval.

Parses annotation JSONs, loads DINOv3 features, and builds
train/test splits at the video level.
"""

import json
import os
import random
from pathlib import Path

import numpy as np

ANNO_DIR = Path("/home/ubuntu/yifan/code/motion-data-process/manual_label/anno/good_quality_round_annotated_634_0312")
FEAT_DIR = Path("/home/ubuntu/yifan/code/FACT_actseg/Data_Filtering/filtered_videos_feat")

# All 16 action classes sorted
ALL_LABELS = [
    "align", "call for action", "clean hand", "close bets", "dealer hits",
    "discard", "double", "hit", "initial hands 1st", "initial hands 2nd",
    "refer", "reveal hole card", "split", "tap", "wait for bets", "wave",
]
LABEL2ID = {l: i for i, l in enumerate(ALL_LABELS)}


def load_all_videos(anno_dir: Path = ANNO_DIR, feat_dir: Path = FEAT_DIR):
    """
    Returns a list of dicts, one per video that has both annotation and feature file:
        {
            "video_id": str,           # e.g. "2025-10-01_06-05-15_000484_003464"
            "segments": [              # from timeline_segments
                {"start_frame": int, "end_frame": int, "label": str}
            ],
            "features": np.ndarray,   # shape (T, 768)
        }
    """
    videos = []
    anno_files = sorted(anno_dir.glob("*_annotations.json"))
    missing_feat = 0

    for af in anno_files:
        video_id = af.name.replace("_annotations.json", "")
        feat_path = feat_dir / f"{video_id}.npy"
        if not feat_path.exists():
            missing_feat += 1
            continue

        with open(af) as f:
            anno = json.load(f)

        segments = []
        for seg in anno["timeline_segments"]:
            label = seg["labels"][0] if seg["labels"] else None
            if label is None or label not in LABEL2ID:
                continue
            segments.append({
                "start_frame": seg["start_frame"],
                "end_frame": seg["end_frame"],
                "label": label,
            })

        if not segments:
            continue

        features = np.load(feat_path)  # (T, 768)
        videos.append({
            "video_id": video_id,
            "segments": segments,
            "features": features,
        })

    print(f"Loaded {len(videos)} videos  |  missing features: {missing_feat}")
    return videos


def build_frame_dataset(videos, stride: int = 5):
    """
    Build a flat list of labeled frames from annotated segments.

    stride: sample every N-th frame within each segment (to reduce redundancy).

    Returns:
        list of dicts:
            {"video_id": str, "frame_idx": int, "label": str,
             "label_id": int, "feat": np.ndarray shape (768,)}
    """
    frames = []
    for v in videos:
        T = v["features"].shape[0]
        for seg in v["segments"]:
            s, e = seg["start_frame"], min(seg["end_frame"], T - 1)
            if s > e:
                continue
            indices = list(range(s, e + 1, stride))
            if not indices:
                indices = [s]
            for idx in indices:
                frames.append({
                    "video_id": v["video_id"],
                    "frame_idx": idx,
                    "label": seg["label"],
                    "label_id": LABEL2ID[seg["label"]],
                    "feat": v["features"][idx],
                })
    return frames


def build_segment_dataset(videos):
    """
    Build a flat list of labeled segments using mean-pooled features.

    Returns same schema as build_frame_dataset but feat = mean over segment.
    """
    segments = []
    for v in videos:
        T = v["features"].shape[0]
        for seg in v["segments"]:
            s, e = seg["start_frame"], min(seg["end_frame"], T - 1)
            if s > e:
                continue
            chunk = v["features"][s:e + 1]
            if chunk.shape[0] == 0:
                continue
            feat = chunk.mean(axis=0)
            segments.append({
                "video_id": v["video_id"],
                "frame_idx": (s + e) // 2,
                "label": seg["label"],
                "label_id": LABEL2ID[seg["label"]],
                "feat": feat,
            })
    return segments


def filter_videos_with_label(videos, label: str):
    """Keep only videos that contain at least one segment with the given label."""
    return [v for v in videos if any(s["label"] == label for s in v["segments"])]


def build_temporal_segment_dataset(videos, n_bins: int = 3):
    """
    Segment-level descriptor using temporal bin concatenation.

    Each segment is divided into n_bins equal-duration windows.
    The per-bin mean features are concatenated, giving a (n_bins * 768)-dim vector
    that preserves temporal ordering (start → mid → end appearance).

    This is strictly more informative than mean-pool when the action has
    a temporal structure (e.g. split: approach → place → separate).

    Returns same schema as build_segment_dataset but feat.shape = (n_bins * 768,).
    """
    segments = []
    for v in videos:
        T = v["features"].shape[0]
        for seg in v["segments"]:
            s, e = seg["start_frame"], min(seg["end_frame"], T - 1)
            if s > e:
                continue
            chunk = v["features"][s:e + 1]  # (L, 768)
            if chunk.shape[0] == 0:
                continue

            # Split into n_bins; if segment shorter than n_bins, replicate last frame
            L = chunk.shape[0]
            bin_feats = []
            for b in range(n_bins):
                b_start = int(b * L / n_bins)
                b_end = int((b + 1) * L / n_bins)
                if b_start >= b_end:
                    # very short segment – reuse last computed bin
                    bin_feats.append(bin_feats[-1] if bin_feats else chunk[0])
                else:
                    bin_feats.append(chunk[b_start:b_end].mean(axis=0))

            feat = np.concatenate(bin_feats, axis=0)  # (n_bins * 768,)
            segments.append({
                "video_id": v["video_id"],
                "video_name": v["video_id"] + ".mp4",
                "start_frame": s,
                "end_frame": e,
                "frame_idx": (s + e) // 2,
                "label": seg["label"],
                "label_id": LABEL2ID[seg["label"]],
                "feat": feat,
                "n_bins": n_bins,
            })
    return segments


def build_augmented_temporal_dataset(videos, n_bins=5, n_augments=3,
                                     jitter=0.15, seed=42):
    """
    Temporal segment dataset with offline augmentation via boundary jitter.
    Each segment → 1 original + n_augments jittered copies.
    jitter: randomly shift start/end by ±jitter * segment_length.
    """
    rng = np.random.default_rng(seed)
    segments = []
    for v in videos:
        T = v["features"].shape[0]
        for seg in v["segments"]:
            s, e = seg["start_frame"], min(seg["end_frame"], T - 1)
            if s >= e:
                continue
            L = e - s
            bounds = [(s, e)]
            for _ in range(n_augments):
                ds = int(rng.uniform(-jitter, jitter) * L)
                de = int(rng.uniform(-jitter, jitter) * L)
                ns, ne = max(0, s + ds), min(T - 1, e + de)
                if ns < ne:
                    bounds.append((ns, ne))

            for bs, be in bounds:
                chunk = v["features"][bs:be + 1]
                if chunk.shape[0] == 0:
                    continue
                cL = chunk.shape[0]
                bin_feats = []
                for b in range(n_bins):
                    b_s = int(b * cL / n_bins)
                    b_e = int((b + 1) * cL / n_bins)
                    if b_s >= b_e:
                        bin_feats.append(bin_feats[-1] if bin_feats else chunk[0])
                    else:
                        bin_feats.append(chunk[b_s:b_e].mean(axis=0))
                feat = np.concatenate(bin_feats, axis=0)
                segments.append({
                    "video_id": v["video_id"],
                    "video_name": v["video_id"] + ".mp4",
                    "start_frame": bs,
                    "end_frame": be,
                    "label": seg["label"],
                    "label_id": LABEL2ID[seg["label"]],
                    "feat": feat,
                    "n_bins": n_bins,
                })
    return segments


def build_window_dataset(videos, window=60, stride=15, n_bins=5,
                         min_overlap_frac=0.3):
    """
    Build training data from sliding windows — matches inference distribution exactly.

    Each window is labeled based on overlap with annotated segments:
      - "split" if ≥ min_overlap_frac of window overlaps a split segment
      - Otherwise the label with highest overlap
      - "discard" if no annotation covers ≥ min_overlap_frac of window
    """
    discard_id = LABEL2ID["discard"]
    min_overlap = int(window * min_overlap_frac)
    items = []

    for v in videos:
        feat = v["features"]       # (T, 768)
        T = feat.shape[0]
        segments = v["segments"]

        for w_start in range(0, T - window + 1, stride):
            w_end = w_start + window

            # Compute overlap with each annotated segment
            overlaps = {}           # label → total overlap frames
            for seg in segments:
                s = seg["start_frame"]
                e = min(seg["end_frame"], T)
                overlap = max(0, min(w_end, e) - max(w_start, s))
                if overlap > 0:
                    lbl = seg["label"]
                    overlaps[lbl] = overlaps.get(lbl, 0) + overlap

            # Assign label — split gets priority when overlap is significant
            if not overlaps:
                label = "discard"
            elif overlaps.get("split", 0) >= min_overlap:
                label = "split"
            else:
                best = max(overlaps, key=overlaps.get)
                label = best if overlaps[best] >= min_overlap else "discard"

            # Temporal bins
            chunk = feat[w_start:w_end]
            cL = chunk.shape[0]
            bin_feats = []
            for b in range(n_bins):
                b_s = int(b * cL / n_bins)
                b_e = int((b + 1) * cL / n_bins)
                if b_s >= b_e:
                    bin_feats.append(bin_feats[-1] if bin_feats else chunk[0])
                else:
                    bin_feats.append(chunk[b_s:b_e].mean(axis=0))
            feat_vec = np.concatenate(bin_feats, axis=0)

            items.append({
                "feat":        feat_vec,
                "label":       label,
                "label_id":    LABEL2ID[label],
                "video_id":    v["video_id"],
                "video_name":  v["video_id"] + ".mp4",
                "start_frame": w_start,
                "end_frame":   w_end,
                "n_bins":      n_bins,
            })

    return items


def train_test_split(videos, test_ratio: float = 0.2, seed: int = 42):
    """
    Split videos 80/20 at the video level.
    Returns (train_videos, test_videos).
    """
    rng = random.Random(seed)
    shuffled = videos[:]
    rng.shuffle(shuffled)
    n_test = max(1, int(len(shuffled) * test_ratio))
    return shuffled[n_test:], shuffled[:n_test]


def get_feature_matrix(items):
    """Stack feat vectors into (N, 768) L2-normalized matrix."""
    feats = np.stack([it["feat"] for it in items])
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-8, norms)
    return (feats / norms).astype(np.float32)


if __name__ == "__main__":
    videos = load_all_videos()
    train_vids, test_vids = train_test_split(videos)
    print(f"Train videos: {len(train_vids)}  |  Test videos: {len(test_vids)}")

    train_frames = build_frame_dataset(train_vids, stride=5)
    test_frames = build_frame_dataset(test_vids, stride=5)
    print(f"Train frames: {len(train_frames)}  |  Test frames: {len(test_frames)}")

    from collections import Counter
    train_dist = Counter(f["label"] for f in train_frames)
    test_dist = Counter(f["label"] for f in test_frames)
    print("\nLabel distribution (train | test):")
    for lbl in ALL_LABELS:
        print(f"  {lbl:25s}  {train_dist.get(lbl, 0):6d}  |  {test_dist.get(lbl, 0):6d}")
