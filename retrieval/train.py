"""
Train ActionDetector (MLP + classifier) with joint CE + triplet loss.

v3 — window-level training (matches inference distribution):
  - Trains on sliding windows from annotated videos (same window/stride as inference)
  - Each window labeled by overlap with annotated segments
  - Eliminates domain gap between training segments and inference windows
  - MLP encoder + multi-class CE + triplet loss
  - Balanced sampling, feature dropout

Supports any action label in ALL_LABELS. Window/stride auto-computed from
the target action's segment length distribution if not specified.

Usage:
    python train.py --action split
    python train.py --action "clean hand"
    python train.py --action hit --window 40 --stride 10   # manual override
    python train.py --action split --no_cv
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

CKPT_DIR = Path(__file__).parent / "checkpoints"
SEED     = 42
N_BINS   = 5       # temporal bins


def auto_window_stride(videos, action: str):
    """Compute window and stride from the action's segment length distribution.

    window ≈ 75th percentile of segment lengths (rounded up to nearest 5)
    stride ≈ window // 4
    """
    lengths = []
    for v in videos:
        T = v["features"].shape[0]
        for s in v["segments"]:
            if s["label"] == action:
                length = min(s["end_frame"], T) - s["start_frame"]
                if length > 0:
                    lengths.append(length)
    if not lengths:
        raise ValueError(f"No segments found for action '{action}'")
    lengths = np.array(lengths)
    p75 = int(np.percentile(lengths, 75))
    window = max(10, int(np.ceil(p75 / 5) * 5))  # round up to nearest 5
    stride = max(2, window // 4)
    print(f"  Auto window/stride for '{action}': "
          f"median={np.median(lengths):.0f}  p75={p75}  "
          f"→ window={window}  stride={stride}")
    return window, stride


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


def ckpt_path(action: str, embed_dim: int, epochs: int, margin: float,
              seed: int) -> Path:
    tag = action.replace(" ", "_")
    return CKPT_DIR / f"detector_{tag}_d{embed_dim}_e{epochs}_m{margin}_s{seed}.pt"


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

def stratified_kfold_videos(all_videos: list, action: str,
                            k: int = 5, seed: int = SEED):
    """Stratified k-fold at video level: ensures action-containing videos
    are evenly distributed across folds."""
    rng = np.random.default_rng(seed)
    action_vids = [v for v in all_videos if any(
        s.get("label") == action for s in v.get("segments", [])
    )]
    other_vids = [v for v in all_videos if v not in action_vids]

    rng.shuffle(action_vids)
    rng.shuffle(other_vids)

    action_folds = [action_vids[i::k] for i in range(k)]
    other_folds = [other_vids[i::k] for i in range(k)]

    folds = []
    for i in range(k):
        val   = action_folds[i] + other_folds[i]
        train = []
        for j in range(k):
            if j != i:
                train += action_folds[j] + other_folds[j]
        folds.append((train, val))
    return folds


# ─────────────────────────────────────────────────────────────────────────────
# Classifier evaluation (P/R/F1 at various thresholds)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_classifier(model, val_items, focus_label,
                        thresholds=(0.5, 0.7, 0.8, 0.9, 0.95)):
    """Evaluate action detection P/R/F1 on validation windows."""
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

def train_and_eval(train_vids, val_vids, action: str, n_bins: int,
                   window: int, stride: int,
                   embed_dim: int, epochs: int, lr: float,
                   margin: float, batch_size: int,
                   triplet_weight: float, feat_dropout: float,
                   fold_idx: int = None, seed: int = SEED):
    """Train an ActionDetector on sliding windows, evaluate on val windows."""
    tr_items = build_window_dataset(
        train_vids, window=window, stride=stride, n_bins=n_bins,
        min_overlap_frac=0.5, focus_label=action)
    va_items = build_window_dataset(
        val_vids, window=window, stride=stride, n_bins=n_bins,
        min_overlap_frac=0.5, focus_label=action)

    action_id = LABEL2ID[action]
    n_tr_action = sum(1 for it in tr_items if it["label_id"] == action_id)
    n_va_action = sum(1 for it in va_items if it["label_id"] == action_id)

    if n_va_action == 0:
        return None

    in_dim = tr_items[0]["feat"].shape[0]
    n_classes = len(ALL_LABELS)

    model = train_detector(
        tr_items, in_dim=in_dim, embed_dim=embed_dim, n_classes=n_classes,
        epochs=epochs, lr=lr, margin=margin, batch_size=batch_size,
        triplet_weight=triplet_weight, feat_dropout=feat_dropout,
        focus_label=action,
    )

    results = evaluate_classifier(model, va_items, focus_label=action)

    prefix = f"Fold {fold_idx}" if fold_idx is not None else "Final"
    print(f"  {prefix:8s}  "
          f"acc={results['accuracy']:.3f}  "
          f"P@0.9={results['P@0.9']:.3f}  R@0.9={results['R@0.9']:.3f}  "
          f"F1@0.9={results['F1@0.9']:.3f}  "
          f"(n_{action}_win={n_va_action}/{results['n_val']})")

    return results, model


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def train(action: str = "split", embed_dim: int = 256, epochs: int = 300,
          lr: float = 3e-3, margin: float = 0.3, batch_size: int = 64,
          triplet_weight: float = 0.5, feat_dropout: float = 0.15,
          n_bins: int = N_BINS, window: int = None, stride: int = None,
          seed: int = SEED, k_folds: int = 5, run_cv: bool = True):

    if action not in LABEL2ID:
        raise ValueError(f"Unknown action '{action}'. Choose from: {ALL_LABELS}")

    set_seeds(seed)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ckpt_path(action, embed_dim, epochs, margin, seed)

    if out_path.exists():
        print(f"Checkpoint already exists: {out_path}")
        print("Delete it to retrain.")
        return out_path

    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"Action: {action}")

    all_videos   = load_all_videos()
    action_videos = filter_videos_with_label(all_videos, action)
    print(f"Videos containing '{action}': {len(action_videos)}")

    # Auto-compute window/stride from segment lengths if not specified
    if window is None or stride is None:
        auto_w, auto_s = auto_window_stride(all_videos, action)
        window = window or auto_w
        stride = stride or auto_s

    print(f"Seed={seed}  embed_dim={embed_dim}  epochs={epochs}  "
          f"margin={margin}  n_bins={n_bins}")
    print(f"Window={window}  stride={stride}  "
          f"triplet_weight={triplet_weight}  feat_dropout={feat_dropout}")

    # ── Cross-validation ─────────────────────────────────────────────────────
    if run_cv:
        print(f"\n── {k_folds}-fold stratified CV (window-level) ──")
        folds = stratified_kfold_videos(action_videos, action=action,
                                        k=k_folds, seed=seed)
        cv_results = []
        for i, (tr_vids, va_vids) in enumerate(folds):
            set_seeds(seed + i)
            ret = train_and_eval(
                tr_vids, va_vids, action=action, n_bins=n_bins,
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

    # ── Final model — train on ALL annotated videos ──────────────────────────
    # Non-action videos provide diverse negatives from the same domain.
    print(f"\n── Training final model (all {len(all_videos)} annotated videos) ──")
    set_seeds(seed)
    all_items = build_window_dataset(
        all_videos, window=window, stride=stride, n_bins=n_bins,
        min_overlap_frac=0.5, focus_label=action)

    c = Counter(it["label"] for it in all_items)
    print(f"  Window labels: {dict(c)}")
    print(f"  Total windows: {len(all_items)}")

    in_dim   = all_items[0]["feat"].shape[0]
    n_classes = len(ALL_LABELS)

    model = train_detector(
        all_items, in_dim=in_dim, embed_dim=embed_dim, n_classes=n_classes,
        epochs=epochs, lr=lr, margin=margin, batch_size=batch_size,
        triplet_weight=triplet_weight, feat_dropout=feat_dropout,
        focus_label=action,
    )

    # Evaluate on training windows (sanity check)
    train_results = evaluate_classifier(model, all_items, focus_label=action)
    print(f"  Train acc={train_results['accuracy']:.3f}  "
          f"P@0.9={train_results['P@0.9']:.3f}  "
          f"R@0.9={train_results['R@0.9']:.3f}")

    # Build gallery for visualisation at inference
    action_items = [it for it in all_items if it["label"] == action]
    gallery_feats = project_items(model, action_items)
    gallery_npy = out_path.with_suffix(".gallery.npy")
    np.save(gallery_npy, gallery_feats)

    train_video_names = [v["video_name"] for v in all_videos]
    torch.save({
        "model_state":  model.state_dict(),
        "model_type":   "SplitDetector",
        "in_dim":       in_dim,
        "embed_dim":    embed_dim,
        "n_classes":    n_classes,
        "n_bins":       n_bins,
        "action":       action,
        "gallery_items": _to_python(action_items),
        "train_video_names": train_video_names,
        "hparams": dict(epochs=epochs, lr=lr, margin=margin,
                        batch_size=batch_size, triplet_weight=triplet_weight,
                        feat_dropout=feat_dropout, seed=seed,
                        window=window, stride=stride),
    }, out_path)

    print(f"\nCheckpoint:    {out_path}")
    print(f"Gallery:       {gallery_npy}  ({len(action_items)} windows)")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action",         type=str,   default="split",
                        help=f"Target action label. Choices: {ALL_LABELS}")
    parser.add_argument("--embed_dim",      type=int,   default=256)
    parser.add_argument("--epochs",         type=int,   default=300)
    parser.add_argument("--lr",             type=float, default=3e-3)
    parser.add_argument("--margin",         type=float, default=0.3)
    parser.add_argument("--batch_size",     type=int,   default=64)
    parser.add_argument("--triplet_weight", type=float, default=0.5)
    parser.add_argument("--feat_dropout",   type=float, default=0.15)
    parser.add_argument("--n_bins",         type=int,   default=N_BINS)
    parser.add_argument("--window",         type=int,   default=None,
                        help="Sliding window size (auto from data if omitted)")
    parser.add_argument("--stride",         type=int,   default=None,
                        help="Sliding window stride (auto from data if omitted)")
    parser.add_argument("--seed",           type=int,   default=SEED)
    parser.add_argument("--folds",          type=int,   default=5)
    parser.add_argument("--no_cv",          action="store_true")
    args = parser.parse_args()

    train(action=args.action, embed_dim=args.embed_dim, epochs=args.epochs,
          lr=args.lr, margin=args.margin, batch_size=args.batch_size,
          triplet_weight=args.triplet_weight, feat_dropout=args.feat_dropout,
          n_bins=args.n_bins, window=args.window, stride=args.stride,
          seed=args.seed, k_folds=args.folds, run_cv=not args.no_cv)
