"""
Sliding-window split detection on new, unannotated videos.

v2 — classifier-based scoring:
  1. Load trained SplitDetector (MLP + classifier).
  2. Slide a fixed-length window over each new video.
  3. Compute P(split) from the classifier for each window.
  4. Pick top-N windows (NMS) above confidence threshold.
  5. Extract side-by-side H.264 clips for high-confidence detections.

Usage:
    python inference.py --feat_dir DATA/feat/ --video_dir DATA/video/
    python inference.py --conf_threshold 0.9
"""

import argparse
import json
import subprocess
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import LABEL2ID
from triplet_retrieval import SplitDetector, ProjectionHead

# ─────────────────────────────────────────────────────────────────────────────
FEAT_DIR  = Path("/home/ubuntu/yifan/code/FACT_actseg/Data_Filtering/filtered_videos_feat")
VIDEO_DIR = Path("/home/ubuntu/yifan/code/FACT_actseg/Data_Filtering/filtered_videos")
OUT_DIR     = Path(__file__).parent / "vis_out" / "inference"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONF_THRESHOLD = 0.9     # P(split) threshold — clean probability, [0,1]

# ─────────────────────────────────────────────────────────────────────────────
# Temporal feature helpers
# ─────────────────────────────────────────────────────────────────────────────

def temporal_bins(feat_seq: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """(T, D) → (n_bins*D,) by equal-duration mean pooling."""
    T, D = feat_seq.shape
    edges = np.linspace(0, T, n_bins + 1, dtype=int)
    parts = [feat_seq[edges[i]:edges[i+1]].mean(axis=0) for i in range(n_bins)]
    return np.concatenate(parts).astype(np.float32)


def sliding_windows(feat: np.ndarray, window: int, stride: int, n_bins: int = 5):
    """Yield (start, end, feat_vec) for every window in feat."""
    T = feat.shape[0]
    for start in range(0, T - window + 1, stride):
        end = start + window
        yield start, end, temporal_bins(feat[start:end], n_bins)


# ─────────────────────────────────────────────────────────────────────────────
# Non-maximum suppression on 1-D score timeline
# ─────────────────────────────────────────────────────────────────────────────

def nms_1d(scores: np.ndarray, starts: np.ndarray, ends: np.ndarray,
           topn: int, iou_thresh: float = 0.3):
    """Return indices of selected windows after greedy NMS."""
    order = np.argsort(-scores)
    selected = []
    suppressed = np.zeros(len(scores), dtype=bool)
    for idx in order:
        if suppressed[idx]:
            continue
        selected.append(idx)
        if len(selected) >= topn:
            break
        s1, e1 = starts[idx], ends[idx]
        for j in range(len(scores)):
            if suppressed[j]:
                continue
            s2, e2 = starts[j], ends[j]
            inter = max(0, min(e1, e2) - max(s1, s2))
            union = (e1 - s1) + (e2 - s2) - inter
            if union > 0 and inter / union > iou_thresh:
                suppressed[j] = True
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Build gallery: split-only items from training set, projected
# ─────────────────────────────────────────────────────────────────────────────

def build_split_gallery(model, train_items):
    """Return (gallery_feats, gallery_items) for split class only."""
    from triplet_retrieval import project_items
    split_id = LABEL2ID["split"]
    split_items = [it for it in train_items if it["label_id"] == split_id]
    projs = project_items(model, split_items)
    return projs, split_items


# ─────────────────────────────────────────────────────────────────────────────
# Score a single new video
# ─────────────────────────────────────────────────────────────────────────────

def score_video(feat_path: Path, model, gallery_feats: np.ndarray = None,
                window: int = 60, stride: int = 15, n_bins: int = 5):
    """
    Returns:
        starts, ends  : (N,) int arrays
        scores        : (N,) float — P(split) from classifier (or cosine sim for legacy)
        best_gal_idx  : (N,) int   — index of closest gallery item per window (or None)
        win_proj      : (N, embed_dim) projected window features
    """
    model.eval()
    feat = np.load(feat_path)   # (T, 768)

    starts, ends, raw_feats = [], [], []
    for s, e, wf in sliding_windows(feat, window, stride, n_bins):
        starts.append(s); ends.append(e); raw_feats.append(wf)

    if len(raw_feats) == 0:
        empty = np.array([])
        return empty, empty, empty, None, np.zeros((0, 1))

    raw_np = np.stack(raw_feats).astype(np.float32)

    split_id = LABEL2ID["split"]
    all_probs = []
    all_emb = []
    is_detector = isinstance(model, SplitDetector)

    for i in range(0, len(raw_np), 512):
        chunk = torch.tensor(raw_np[i:i+512], dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            if is_detector:
                emb, logits = model(chunk)
                probs = torch.softmax(logits, dim=-1)
                all_probs.append(
                    np.array(probs[:, split_id].tolist(), dtype=np.float32))
                all_emb.append(
                    np.array(emb.tolist(), dtype=np.float32))
            else:
                # Legacy: ProjectionHead
                proj = model(chunk)
                all_emb.append(
                    np.array(proj.tolist(), dtype=np.float32))

    win_proj = np.concatenate(all_emb, axis=0)

    if is_detector:
        scores = np.concatenate(all_probs)
    else:
        # Legacy fallback: cosine sim to gallery
        if gallery_feats is not None:
            sims = win_proj @ gallery_feats.T
            scores = sims.max(axis=1)
        else:
            scores = np.zeros(len(win_proj))

    # Gallery match for visualization
    best_gal_idx = None
    if gallery_feats is not None:
        sims = win_proj @ gallery_feats.T
        best_gal_idx = sims.argmax(axis=1)

    return (np.array(starts), np.array(ends),
            scores, best_gal_idx, win_proj)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

GALLERY_VIDEO_DIR = Path(
    "/home/ubuntu/yifan/code/FACT_actseg/Data_Filtering/filtered_videos"
)


def find_gallery_video(video_name: str):
    """Locate a gallery video by stem name."""
    for candidate in [
        GALLERY_VIDEO_DIR / video_name,
        GALLERY_VIDEO_DIR / (video_name + ".mp4"),
    ]:
        if candidate.exists():
            return candidate
    return None


def get_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return fps


def extract_clip_ffmpeg(src: Path, start_frame: int, end_frame: int,
                        dst: Path, fps: float = 30.0, pad_frames: int = 0):
    s = max(0, start_frame - pad_frames) / fps
    dur = (end_frame - start_frame + 2 * pad_frames) / fps
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = (
        f'ffmpeg -y -ss {s:.4f} -i "{src}" -t {dur:.4f} '
        f'-c:v libx264 -preset fast -crf 18 -an '
        f'"{dst}" -loglevel error'
    )
    subprocess.run(cmd, shell=True, check=True)


def make_sidebyside_ffmpeg(query_clip: Path, gallery_clip: Path,
                           out_path: Path, label_text: str = ""):
    drawtext = (
        f"drawtext=text='QUERY':x=10:y=10:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5,"
        f"drawtext=text='GALLERY':x=w/2+10:y=10:fontsize=24:fontcolor=yellow:box=1:boxcolor=black@0.5"
    )
    if label_text:
        drawtext += (
            f",drawtext=text='{label_text}':x=(w-text_w)/2:y=h-30"
            f":fontsize=18:fontcolor=white:box=1:boxcolor=black@0.6"
        )
    cmd = (
        f'ffmpeg -y -i "{query_clip}" -i "{gallery_clip}" '
        f'-filter_complex "[0:v]scale=640:-2[q];[1:v]scale=640:-2[g];'
        f'[q][g]hstack[out];[out]{drawtext}[v]" '
        f'-map "[v]" -c:v libx264 -preset fast -crf 18 -an '
        f'"{out_path}" -loglevel error'
    )
    subprocess.run(cmd, shell=True, check=True)


def visualize_timeline(
    video_id: str,
    starts: np.ndarray, ends: np.ndarray,
    scores: np.ndarray, det_indices: list,
    total_frames: int,
    save_dir: Path = OUT_DIR,
):
    """Plot P(split) timeline with detections highlighted."""
    save_dir.mkdir(parents=True, exist_ok=True)
    mid = (starts + ends) / 2

    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.plot(mid, scores, color="#5b9bd5", lw=0.9, alpha=0.7, label="P(split)")
    ax.fill_between(mid, scores, alpha=0.18, color="#5b9bd5")

    thresh = scores[det_indices].min() if det_indices else scores.max()
    ax.axhline(thresh, color="#ed7d31", ls="--", lw=1,
               label=f"detection threshold ({thresh:.3f})")

    for rank, di in enumerate(det_indices):
        ax.axvspan(starts[di], ends[di], alpha=0.28, color="#c00000",
                   label="detection" if rank == 0 else None)
        ax.text(mid[di], scores[di] + 0.015, f"#{rank+1}", ha="center",
                fontsize=7.5, color="#c00000", fontweight="bold")

    ax.set_xlim(0, total_frames)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("frame index")
    ax.set_ylabel("P(split)")
    ax.set_title(f"Split detection timeline — {video_id}")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    out_path = save_dir / f"{video_id}_timeline.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Clip extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_clips(
    video_id: str,
    starts: np.ndarray, ends: np.ndarray,
    scores: np.ndarray, best_gal_idx,
    gallery_items: list,
    det_indices: list,
    conf_threshold: float = CONF_THRESHOLD,
    pad_frames: int = 15,
    save_dir: Path = OUT_DIR,
    video_dir: Path = None,
):
    """
    For each high-confidence detection (score >= conf_threshold):
      1. Extract the query clip
      2. Extract the matched gallery clip
      3. Stitch side-by-side
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    video_dir = video_dir or VIDEO_DIR
    query_video = video_dir / f"{video_id}.mp4"
    if not query_video.exists():
        print(f"    [skip clips] query video not found: {query_video.name}")
        return

    fps = get_fps(query_video)
    tmp_dir = save_dir / "_tmp"
    tmp_dir.mkdir(exist_ok=True)

    n_saved = 0
    for rank, di in enumerate(det_indices):
        score = float(scores[di])
        if score < conf_threshold:
            print(f"    [skip] #{rank+1} P(split)={score:.3f} < {conf_threshold}")
            continue

        s, e = int(starts[di]), int(ends[di])
        stem = f"{video_id}_det{rank+1:02d}_p{score:.3f}"

        # Extract query clip
        q_tmp = tmp_dir / f"{stem}_query.mp4"
        extract_clip_ffmpeg(query_video, s, e, q_tmp, fps=fps,
                            pad_frames=pad_frames)

        if best_gal_idx is not None and gallery_items:
            g_item = gallery_items[int(best_gal_idx[di])]
            g_vid_name = g_item["video_name"].replace(".mp4", "")
            g_start, g_end = g_item["start_frame"], g_item["end_frame"]
            gallery_video = find_gallery_video(g_vid_name + ".mp4")

            if gallery_video:
                g_fps = get_fps(gallery_video)
                g_tmp = tmp_dir / f"{stem}_gallery.mp4"
                extract_clip_ffmpeg(gallery_video, g_start, g_end, g_tmp,
                                    fps=g_fps, pad_frames=pad_frames)
                label = f"P(split)={score:.3f}  |  gallery: {g_item['label']}"
                out_path = save_dir / f"{stem}_sidebyside.mp4"
                make_sidebyside_ffmpeg(q_tmp, g_tmp, out_path,
                                       label_text=label)
                print(f"    Saved clip: {out_path.name}")
            else:
                out_path = save_dir / f"{stem}_query_only.mp4"
                shutil.copy(q_tmp, out_path)
                print(f"    Saved clip (query only): {out_path.name}")
        else:
            out_path = save_dir / f"{stem}_query_only.mp4"
            shutil.copy(q_tmp, out_path)
            print(f"    Saved clip (query only): {out_path.name}")

        n_saved += 1

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"    {n_saved} clip(s) saved  "
          f"({len(det_indices)-n_saved} below threshold {conf_threshold})")


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint loading
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(ckpt: Path):
    """Load model and gallery from checkpoint. Returns (model, gallery_feats, gallery_items, n_bins)."""
    data = torch.load(ckpt, map_location="cpu", weights_only=False)
    model_type = data.get("model_type", "ProjectionHead")

    if model_type == "SplitDetector":
        model = SplitDetector(data["in_dim"], data["embed_dim"], data["n_classes"])
    else:
        model = ProjectionHead(data["in_dim"], data.get("out_dim", data.get("embed_dim")))

    model.load_state_dict(data["model_state"])
    model.to(DEVICE).eval()

    n_bins = data.get("n_bins", 3)
    gallery_feats = np.load(ckpt.with_suffix(".gallery.npy"))

    train_video_ids = set(data.get("train_video_ids", []))

    print(f"Loaded checkpoint: {ckpt.name}  (type={model_type})")
    print(f"  hparams: {data['hparams']}")
    print(f"  n_bins={n_bins}  split gallery: {len(data['gallery_items'])} segments")
    if train_video_ids:
        print(f"  train videos: {len(train_video_ids)} (will skip at inference)")
    return model, gallery_feats, data["gallery_items"], n_bins, train_video_ids


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(ckpt: Path, window: int = 60, stride: int = 15, topn: int = 1,
        feat_dir: Path = None, video_dir: Path = None,
        conf_threshold: float = CONF_THRESHOLD):

    feat_dir  = feat_dir  or FEAT_DIR
    video_dir = video_dir or VIDEO_DIR
    print(f"Device: {DEVICE}")
    print(f"Feat dir:  {feat_dir}")
    print(f"Video dir: {video_dir}")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    print("\n[1/3] Loading checkpoint...")
    model, gallery_feats, gallery_items, n_bins, train_ids = load_checkpoint(ckpt)

    # ── Score new videos ─────────────────────────────────────────────────────
    print(f"\n[2/3] Scoring new videos (window={window}, stride={stride}, "
          f"n_bins={n_bins})...")
    feat_files = sorted(feat_dir.glob("*.npy"))
    all_detections = {}
    n_skipped_train = 0

    for feat_path in feat_files:
        video_id = feat_path.stem

        # Skip training videos to avoid query=gallery
        if train_ids and video_id in train_ids:
            n_skipped_train += 1
            continue

        print(f"\n  {video_id}")

        starts, ends, scores, best_gal_idx, win_proj = score_video(
            feat_path, model, gallery_feats,
            window=window, stride=stride, n_bins=n_bins
        )

        if len(scores) == 0:
            print(f"    No windows (video too short)")
            continue

        det_indices = nms_1d(scores, starts, ends, topn=topn)

        print(f"    Windows: {len(scores)}  |  Detections: {len(det_indices)}")
        for rank, di in enumerate(det_indices):
            gal_info = ""
            if best_gal_idx is not None and gallery_items:
                g = gallery_items[best_gal_idx[di]]
                gal_info = (f"  → {g['video_name']} "
                           f"[{g['start_frame']}–{g['end_frame']}]")
            print(f"    #{rank+1}  frames [{starts[di]}–{ends[di]}]  "
                  f"P(split)={scores[di]:.4f}{gal_info}")

        all_detections[video_id] = {
            "detections": [
                {
                    "rank": rank + 1,
                    "query_start": int(starts[di]),
                    "query_end": int(ends[di]),
                    "score": float(scores[di]),
                    **({"gallery_video": gallery_items[best_gal_idx[di]]["video_name"],
                        "gallery_start": int(gallery_items[best_gal_idx[di]]["start_frame"]),
                        "gallery_end": int(gallery_items[best_gal_idx[di]]["end_frame"]),
                        "gallery_label": gallery_items[best_gal_idx[di]]["label"],
                       } if best_gal_idx is not None and gallery_items else {}),
                }
                for rank, di in enumerate(det_indices)
            ]
        }

        # ── Extract clips ──────────────────────────────────────────────────
        extract_clips(video_id, starts, ends, scores, best_gal_idx,
                      gallery_items, det_indices,
                      conf_threshold=conf_threshold, video_dir=video_dir)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / f"detections_{feat_dir.name}.json"
    with open(json_path, "w") as f:
        json.dump(all_detections, f, indent=2)
    print(f"\nDetections saved: {json_path}")
    if n_skipped_train:
        print(f"Skipped {n_skipped_train} training videos")
    return all_detections


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",      type=Path, default=None,
                        help="Checkpoint .pt — defaults to latest in checkpoints/")
    parser.add_argument("--feat_dir",  type=Path,
                        default=FEAT_DIR,
                        help="Directory of .npy feature files")
    parser.add_argument("--video_dir", type=Path,
                        default=VIDEO_DIR,
                        help="Matching video directory for clip extraction")
    parser.add_argument("--window",    type=int,  default=60)
    parser.add_argument("--stride",    type=int,  default=15)
    parser.add_argument("--topn",           type=int,   default=1)
    parser.add_argument("--conf_threshold", type=float, default=CONF_THRESHOLD,
                        help="Min P(split) to save a clip (0-1)")
    args = parser.parse_args()

    ckpt = args.ckpt
    if ckpt is None:
        ckpt_dir = Path(__file__).parent / "checkpoints"
        candidates = sorted(ckpt_dir.glob("*.pt"))
        if not candidates:
            raise FileNotFoundError(
                "No checkpoint found. Run  python train.py  first.")
        ckpt = candidates[-1]

    run(ckpt=ckpt, window=args.window, stride=args.stride, topn=args.topn,
        feat_dir=args.feat_dir, video_dir=args.video_dir,
        conf_threshold=args.conf_threshold)
