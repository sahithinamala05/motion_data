"""
Track split card identity using DINOv2 feature matching.

For same-suit pairs where visual appearance can't distinguish cards,
extracts DINOv2 patch features around each card at the start frame,
then finds the best feature match at the end frame.

This is simpler than full DINO-Tracker (no per-video training needed).
Just extracts features and matches by cosine similarity.

Usage:
    python split_dino_tracker.py --video_name 2025-10-01_08-04-02_012398_014586 \
                                  --start_frame 1315 --end_frame 1336

Requires: torch, torchvision (for DINOv2)
"""

import argparse
import json
import math
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# ─── Paths ───────────────────────────────────────────────────────────────────
RAW_VIDEO_DIR = "/home/ubuntu/us-west-3-fs/live_dealer_blackjack/raw/batch_01"
ANNOTATIONS_DIR = (
    "/home/ubuntu/us-west-3-fs/sahithi/hard_action_annotations/full_run/split_try_2"
)
IMG_W, IMG_H = 1280, 720
PATCH_SIZE = 14  # DINOv2 patch size
CROP_PAD = 10    # padding around card bbox for feature extraction


# ─── DINOv2 model ────────────────────────────────────────────────────────────

_model = None
_transform = None


def _load_model():
    global _model, _transform
    if _model is not None:
        return _model, _transform
    _model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    _model.eval()
    if torch.cuda.is_available():
        _model = _model.cuda()

    from torchvision import transforms
    _transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return _model, _transform


# ─── Feature extraction ─────────────────────────────────────────────────────

def _crop_card(frame, box, pad=CROP_PAD):
    """Crop card region from frame with padding."""
    h, w = frame.shape[:2]
    x1 = max(0, box[0] - pad)
    y1 = max(0, box[1] - pad)
    x2 = min(w, box[2] + pad)
    y2 = min(h, box[3] + pad)
    return frame[y1:y2, x1:x2]


def _extract_feature(model, transform, frame_bgr, box):
    """Extract DINOv2 CLS token feature for a card crop."""
    crop = _crop_card(frame_bgr, box)
    if crop.size == 0:
        return None
    # resize to 224x224 for DINOv2
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop_resized = cv2.resize(crop_rgb, (224, 224))
    tensor = transform(crop_resized).unsqueeze(0)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    with torch.no_grad():
        features = model(tensor)  # CLS token: (1, 384)
    return features[0].cpu()


def match_cards_by_feature(
    frame_start, frame_end,
    start_cards, end_cards,
):
    """Match end cards to start cards using DINOv2 feature similarity.

    Args:
        frame_start: BGR frame at start
        frame_end: BGR frame at end
        start_cards: list of 2 card dicts with 'box' key (sorted: [upper, lower])
        end_cards: list of 2 card dicts with 'box' key (unsorted)

    Returns:
        (e0, e1) where e0 matches start_cards[0], e1 matches start_cards[1]
    """
    model, transform = _load_model()

    # extract features for start cards
    s0_feat = _extract_feature(model, transform, frame_start, start_cards[0]["box"])
    s1_feat = _extract_feature(model, transform, frame_start, start_cards[1]["box"])

    if s0_feat is None or s1_feat is None:
        return end_cards[0], end_cards[1]

    # extract features for end cards
    ea_feat = _extract_feature(model, transform, frame_end, end_cards[0]["box"])
    eb_feat = _extract_feature(model, transform, frame_end, end_cards[1]["box"])

    if ea_feat is None or eb_feat is None:
        return end_cards[0], end_cards[1]

    # cosine similarity
    sim_s0_ea = F.cosine_similarity(s0_feat.unsqueeze(0), ea_feat.unsqueeze(0)).item()
    sim_s0_eb = F.cosine_similarity(s0_feat.unsqueeze(0), eb_feat.unsqueeze(0)).item()
    sim_s1_ea = F.cosine_similarity(s1_feat.unsqueeze(0), ea_feat.unsqueeze(0)).item()
    sim_s1_eb = F.cosine_similarity(s1_feat.unsqueeze(0), eb_feat.unsqueeze(0)).item()

    # optimal assignment: maximize total similarity
    score_direct = sim_s0_ea + sim_s1_eb   # s0→ea, s1→eb
    score_cross = sim_s0_eb + sim_s1_ea    # s0→eb, s1→ea

    if score_direct >= score_cross:
        return end_cards[0], end_cards[1]  # ea=e0, eb=e1
    else:
        return end_cards[1], end_cards[0]  # eb=e0, ea=e1


# ─── Video helpers ───────────────────────────────────────────────────────────

def _get_frame(video_name_base, frame_idx):
    """Load a specific frame from video."""
    parts = video_name_base.split('_')
    base = f"{parts[0]}_{parts[1]}".replace('_', ' ', 1)
    clip_start = int(parts[2])
    for name in (f"{base}.mp4", f"{parts[0]}_{parts[1]}.mp4"):
        path = os.path.join(RAW_VIDEO_DIR, name)
        if os.path.exists(path):
            break
    else:
        return None
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, (clip_start - 1) + frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    if frame.shape[1] != IMG_W or frame.shape[0] != IMG_H:
        frame = cv2.resize(frame, (IMG_W, IMG_H))
    return frame


# ─── Test on annotation ─────────────────────────────────────────────────────

def test_on_video(video_name, start_frame):
    """Test DINO feature matching on a specific split segment."""
    ann_path = os.path.join(ANNOTATIONS_DIR, f"{video_name}_annotations.json")
    ann = json.load(open(ann_path))

    seg = None
    for s in ann["timeline_segments"]:
        if s["start_frame"] == start_frame and s["bounding_boxes"]:
            seg = s
            break
    if seg is None:
        print(f"No split segment found at frame {start_frame}")
        return

    kfs = seg["bounding_boxes"][0]["keyframes"]
    sf, ef = kfs[0]["frame"], kfs[1]["frame"]

    print(f"Segment: {video_name} f{sf}-{ef}")
    print(f"Start KF: [{kfs[0]['rank'][0]}{kfs[0]['suit'][0]}] [{kfs[0]['rank'][1]}{kfs[0]['suit'][1]}]")
    print(f"End KF:   [{kfs[1]['rank'][0]}{kfs[1]['suit'][0]}] [{kfs[1]['rank'][1]}{kfs[1]['suit'][1]}]")

    # load frames
    frame_start = _get_frame(video_name, sf)
    frame_end = _get_frame(video_name, ef)
    if frame_start is None or frame_end is None:
        print("Could not load frames")
        return

    # build card dicts
    s0 = {"box": kfs[0]["card_box"][0], "rank": kfs[0]["rank"][0], "suit": kfs[0]["suit"][0]}
    s1 = {"box": kfs[0]["card_box"][1], "rank": kfs[0]["rank"][1], "suit": kfs[0]["suit"][1]}
    ea = {"box": kfs[1]["card_box"][0], "rank": kfs[1]["rank"][0], "suit": kfs[1]["suit"][0]}
    eb = {"box": kfs[1]["card_box"][1], "rank": kfs[1]["rank"][1], "suit": kfs[1]["suit"][1]}

    print(f"\nCurrent mapping (from pipeline):")
    print(f"  s0={s0['rank']}{s0['suit']}@box={s0['box']} -> e0={ea['rank']}{ea['suit']}@box={ea['box']}")
    print(f"  s1={s1['rank']}{s1['suit']}@box={s1['box']} -> e1={eb['rank']}{eb['suit']}@box={eb['box']}")

    # DINO matching
    e0, e1 = match_cards_by_feature(frame_start, frame_end, [s0, s1], [ea, eb])

    print(f"\nDINO matching:")
    print(f"  s0={s0['rank']}{s0['suit']} -> e0={e0['rank']}{e0['suit']}@box={e0['box']}")
    print(f"  s1={s1['rank']}{s1['suit']} -> e1={e1['rank']}{e1['suit']}@box={e1['box']}")

    swapped = (e0["box"] != ea["box"])
    print(f"\n{'SWAPPED' if swapped else 'SAME'} as pipeline")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_name", required=True)
    parser.add_argument("--start_frame", type=int, required=True)
    args = parser.parse_args()
    test_on_video(args.video_name, args.start_frame)
