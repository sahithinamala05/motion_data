"""
Train SplitDetector (MLP + classifier) with joint CE + triplet loss.

v3 — window-level training (matches inference distribution):
  - Trains on sliding windows from annotated videos (same window/stride as inference)
  - Each window labeled by overlap with annotated segments
  - Eliminates domain gap between training segments and inference windows
  - MLP encoder + multi-class CE + triplet loss
  - Balanced sampling, feature dropout

Usage:
    python train.py                 # CV + final model
    python train.py --epochs 300
    python train.py --no_cv         # skip CV, just train final model
"""

import argparse
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from dataset import (
    ALL_LABELS, LABEL2ID,
    load_all_videos, filter_videos_with_label,
    build_temporal_segment_dataset,
    build_window_dataset,
)
from triplet_retrieval import SplitDetector, train_detector, project_items
from inference import build_split_gallery

CKPT_DIR = Path(__file__).parent / "checkpoints"
SEED     = 42

N_BINS   = 5       # temporal bins
WINDOW   = 60      # sliding window size (frames) — must match inference
STRIDE   = 15      # sliding window stride — must match inference


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ckpt_path(embed_dim: int, epochs: int, margin: float, seed: int) -> Path:
    return CKPT_DIR / f"detector_v3_d{embed_dim}_e{epochs}_m{margin}_s{seed}.pt"


def _to_python(obj):
    """Recursively convert numpy scalars to plain Python types for torch.save."""
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_python(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return obj.item()
        return obj.tolist()
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# Stratified k-fold split at video level
# ─────────────────────────────────────────────────────────────────────────────

def stratified_kfold_videos(all_videos: list, k: int = 5, seed: int = SEED):
    rng = np.random.default_rng(seed)
    split_vids = [v for v in all_videos if any(
        s.get("label") == "split" for s in v.get("segments", [])
    )]
    other_vids = [v for v in all_videos if v not in split_vids]

    rng.shuffle(split_vids)
    rng.shuffle(other_vids)

    split_folds = [split_vids[i::k] for i in range(k)]
    other_folds = [other_vids[i::k] for i in range(k)]

    folds = []
    for i in range(k):
        val   = split_folds[i] + other_folds[i]
        train = []
        for j in range(k):
            if j != i:
                train += split_folds[j] + other_folds[j]
        folds.append((train, val))
    return folds


# ─────────────────────────────────────────────────────────────────────────────
# Classifier evaluation (P/R/F1 at various thresholds)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_classifier(model, val_items, focus_label="split",
                        thresholds=(0.5, 0.7, 0.8, 0.9, 0.95)):
    """Evaluate split detection P/R/F1 on validation windows."""
    model.eval()
    feats = np.stack([it["feat"] for it in val_items]).astype(np.float32)
    labels = np.array([it["label_id"] for it in val_items])
    focus_id = LABEL2ID[focus_label]

    device = next(model.parameters()).device

    all_probs = []
    for i in range(0, len(feats), 512):
        chunk = torch.tensor(feats[i:i+512], dtype=torch.float32).to(device)
        with torch.no_grad():
            _, logits = model(chunk)
            probs = torch.softmax(logits, dim=-1)
            all_probs.append(np.array(probs.cpu().tolist(), dtype=np.float32))

    probs = np.concatenate(all_probs, axis=0)
    split_probs = probs[:, focus_id]
    pred_labels = probs.argmax(axis=1)

    accuracy = (pred_labels == labels).mean()
    split_mask = (labels == focus_id)
    n_split = int(split_mask.sum())

    results = {"accuracy": float(accuracy), "n_val": len(labels),
               "n_split": n_split}

    for t in thresholds:
        pred_split = split_probs >= t
        tp = int((pred_split & split_mask).sum())
        fp = int((pred_split & ~split_mask).sum())
        fn = int((~pred_split & split_mask).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) \
            if (precision + recall) > 0 else 0.0
        results[f"P@{t}"] = precision
        results[f"R@{t}"] = recall
        results[f"F1@{t}"] = f1

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Single train+eval pass (used by CV and final training)
# ─────────────────────────────────────────────────────────────────────────────

def train_and_eval(train_vids, val_vids, n_bins: int,
                   window: int, stride: int,
                   embed_dim: int, epochs: int, lr: float,
                   margin: float, batch_size: int,
                   triplet_weight: float, feat_dropout: float,
                   fold_idx: int = None, seed: int = SEED):
    """Train a SplitDetector on sliding windows, evaluate on val windows."""
    tr_items = build_window_dataset(
        train_vids, window=window, stride=stride, n_bins=n_bins,
        min_overlap_frac=0.5)
    va_items = build_window_dataset(
        val_vids, window=window, stride=stride, n_bins=n_bins,
        min_overlap_frac=0.5)

    split_id = LABEL2ID["split"]
    n_tr_split = sum(1 for it in tr_items if it["label_id"] == split_id)
    n_va_split = sum(1 for it in va_items if it["label_id"] == split_id)

    if n_va_split == 0:
        return None

    in_dim = tr_items[0]["feat"].shape[0]
    n_classes = len(ALL_LABELS)

    model = train_detector(
        tr_items, in_dim=in_dim, embed_dim=embed_dim, n_classes=n_classes,
        epochs=epochs, lr=lr, margin=margin, batch_size=batch_size,
        triplet_weight=triplet_weight, feat_dropout=feat_dropout,
        focus_label="split",
    )

    results = evaluate_classifier(model, va_items)

    prefix = f"Fold {fold_idx}" if fold_idx is not None else "Final"
    print(f"  {prefix:8s}  "
          f"acc={results['accuracy']:.3f}  "
          f"P@0.9={results['P@0.9']:.3f}  R@0.9={results['R@0.9']:.3f}  "
          f"F1@0.9={results['F1@0.9']:.3f}  "
          f"(n_split_win={n_va_split}/{results['n_val']})")

    return results, model


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def train(embed_dim: int = 256, epochs: int = 300, lr: float = 3e-3,
          margin: float = 0.3, batch_size: int = 64,
          triplet_weight: float = 0.5, feat_dropout: float = 0.15,
          n_bins: int = N_BINS, window: int = WINDOW, stride: int = STRIDE,
          seed: int = SEED, k_folds: int = 5, run_cv: bool = True):

    set_seeds(seed)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ckpt_path(embed_dim, epochs, margin, seed)

    if out_path.exists():
        print(f"Checkpoint already exists: {out_path}")
        print("Delete it to retrain.")
        return out_path

    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"Seed={seed}  embed_dim={embed_dim}  epochs={epochs}  "
          f"margin={margin}  n_bins={n_bins}")
    print(f"Window={window}  stride={stride}  "
          f"triplet_weight={triplet_weight}  feat_dropout={feat_dropout}")

    all_videos   = load_all_videos()
    split_videos = filter_videos_with_label(all_videos, "split")
    print(f"Split-containing videos: {len(split_videos)}")

    # ── Cross-validation ─────────────────────────────────────────────────────
    if run_cv:
        print(f"\n── {k_folds}-fold stratified CV (window-level) ──")
        folds = stratified_kfold_videos(split_videos, k=k_folds, seed=seed)
        cv_results = []
        for i, (tr_vids, va_vids) in enumerate(folds):
            set_seeds(seed + i)
            ret = train_and_eval(
                tr_vids, va_vids, n_bins=n_bins,
                window=window, stride=stride,
                embed_dim=embed_dim, epochs=epochs, lr=lr,
                margin=margin, batch_size=batch_size,
                triplet_weight=triplet_weight, feat_dropout=feat_dropout,
                fold_idx=i + 1, seed=seed + i)
            if ret is not None:
                cv_results.append(ret[0])

        if cv_results:
            for key in ["accuracy", "P@0.9", "R@0.9", "F1@0.9"]:
                vals = [r[key] for r in cv_results]
                print(f"  Mean {key}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")

    # ── Final model — train on ALL 634 annotated videos ─────────────────────
    # 586 non-split videos provide diverse negatives from the same blackjack
    # domain, teaching the model "these common actions are NOT split".
    print(f"\n── Training final model (all {len(all_videos)} annotated videos) ──")
    set_seeds(seed)
    all_items = build_window_dataset(
        all_videos, window=window, stride=stride, n_bins=n_bins,
        min_overlap_frac=0.5)   # stricter: window must be majority-split

    c = Counter(it["label"] for it in all_items)
    print(f"  Window labels: {dict(c)}")
    print(f"  Total windows: {len(all_items)}")

    in_dim   = all_items[0]["feat"].shape[0]
    n_classes = len(ALL_LABELS)

    model = train_detector(
        all_items, in_dim=in_dim, embed_dim=embed_dim, n_classes=n_classes,
        epochs=epochs, lr=lr, margin=margin, batch_size=batch_size,
        triplet_weight=triplet_weight, feat_dropout=feat_dropout,
        focus_label="split",
    )

    # Evaluate on training windows (sanity check)
    train_results = evaluate_classifier(model, all_items)
    print(f"  Train acc={train_results['accuracy']:.3f}  "
          f"P@0.9={train_results['P@0.9']:.3f}  "
          f"R@0.9={train_results['R@0.9']:.3f}")

    # Build split gallery for visualisation at inference
    split_items = [it for it in all_items if it["label"] == "split"]
    gallery_feats = project_items(model, split_items)
    gallery_npy = out_path.with_suffix(".gallery.npy")
    np.save(gallery_npy, gallery_feats)

    train_video_ids = [v["video_id"] for v in all_videos]
    torch.save({
        "model_state":  model.state_dict(),
        "model_type":   "SplitDetector",
        "in_dim":       in_dim,
        "embed_dim":    embed_dim,
        "n_classes":    n_classes,
        "n_bins":       n_bins,
        "gallery_items": _to_python(split_items),
        "train_video_ids": train_video_ids,
        "hparams": dict(epochs=epochs, lr=lr, margin=margin,
                        batch_size=batch_size, triplet_weight=triplet_weight,
                        feat_dropout=feat_dropout, seed=seed,
                        window=window, stride=stride),
    }, out_path)

    print(f"\nCheckpoint:    {out_path}")
    print(f"Split gallery: {gallery_npy}  ({len(split_items)} windows)")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_dim",      type=int,   default=256)
    parser.add_argument("--epochs",         type=int,   default=300)
    parser.add_argument("--lr",             type=float, default=3e-3)
    parser.add_argument("--margin",         type=float, default=0.3)
    parser.add_argument("--batch_size",     type=int,   default=64)
    parser.add_argument("--triplet_weight", type=float, default=0.5)
    parser.add_argument("--feat_dropout",   type=float, default=0.15)
    parser.add_argument("--n_bins",         type=int,   default=N_BINS)
    parser.add_argument("--window",         type=int,   default=WINDOW)
    parser.add_argument("--stride",         type=int,   default=STRIDE)
    parser.add_argument("--seed",           type=int,   default=SEED)
    parser.add_argument("--folds",          type=int,   default=5)
    parser.add_argument("--no_cv",          action="store_true")
    args = parser.parse_args()

    train(embed_dim=args.embed_dim, epochs=args.epochs, lr=args.lr,
          margin=args.margin, batch_size=args.batch_size,
          triplet_weight=args.triplet_weight, feat_dropout=args.feat_dropout,
          n_bins=args.n_bins, window=args.window, stride=args.stride,
          seed=args.seed, k_folds=args.folds, run_cv=not args.no_cv)
