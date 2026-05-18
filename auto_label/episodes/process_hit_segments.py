"""
Process instance_segments.json to extract hit/dealer-hits segments with card detection.

Filtering pipeline (each stage is an independent function):
  Stage 1 – duration      : drop hit/dealer-hits segments < MIN_SEGMENT_FRAMES
  Stage 2 – new card diff : find newly appeared card via greedy frame-diff matching
  Stage 3 – spatial       :
      dealer hits – centroid inside DEALER_BBOX_PX; if multiple, pick highest-x
      hit         – centroid y >= HIT_MIN_Y_PX;     if multiple, pick highest-y

Supports resuming after interruption: already-written annotation JSONs are
skipped on re-run.

Usage:
    python process_hit_segments.py [--vis_count N] [--fail_vis_count N]
    --vis_count       0   no success vis (default); N=random N; -1=all
    --fail_vis_count  0   no failure vis (default); N=random N; -1=all
"""

import argparse
import json
import math
import os
import random
import string
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ─── paths ────────────────────────────────────────────────────────────────────
SEGMENTS_JSON = (
    "/home/ubuntu/us-west-3-fs/sahithi/hard_action_annotations/instance_segments.json"
)
CARDS_DIR = (
    "/home/ubuntu/us-west-3-fs/sahithi/hard_action_annotations/card_detections_roundwise"
)
RAW_VIDEO_DIR = "/home/ubuntu/us-west-3-fs/live_dealer_blackjack/raw/batch_01"
OUTPUT_DIR = Path("/home/ubuntu/us-west-3-fs/sahithi/hard_action_annotations/full_run/hit")
FAILURE_VIS_DIR = OUTPUT_DIR / "failure_vis"
SUCCESS_VIS_DIR = OUTPUT_DIR / "success_vis"

# ─── constants ────────────────────────────────────────────────────────────────
IMG_W, IMG_H = 1280, 720
MIN_SEGMENT_FRAMES = 30
HIT_LABELS = {"hit", "dealer hits"}
MATCH_DIST_THRESH = 120          # px – greedy card-match threshold

# dealer hits: centroid must lie inside this pixel bbox
DEALER_BBOX_PX = ((535, 411), (800, 489))
# hit: centroid y must be >= this value (px)
HIT_MIN_Y_PX = 450


# ═══════════════════════════════════════════════════════════════════════════════
# Low-level helpers
# ═══════════════════════════════════════════════════════════════════════════════

def rand_id(n: int = 10) -> str:
    return "".join(random.choices(string.ascii_letters + string.digits, k=n))


def centre_dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


SCALE = 1280 / 1920  # card detections are 1080p, scale to 720p


def _scale_det(det: dict) -> dict:
    """Scale a single card detection from 1080p to 720p coordinates."""
    cx, cy = det["polygon_center"][0]
    x1, y1, x2, y2 = det["box"]
    return {
        **det,
        "polygon_center": [[cx * SCALE, cy * SCALE]],
        "box": [int(x1 * SCALE), int(y1 * SCALE),
                int(x2 * SCALE), int(y2 * SCALE)],
    }


def load_card_detections(jsonl_path: str) -> Dict[int, List[dict]]:
    """Return {frame_id: [detections]} keeping only frames with detections.
    Coordinates are scaled from 1080p to 720p."""
    out: Dict[int, List[dict]] = {}
    with open(jsonl_path) as f:
        for line in f:
            obj = json.loads(line)
            if obj["detections"]:
                out[obj["frame_id"]] = [_scale_det(d) for d in obj["detections"]]
    return out


def nearest_frame_with_detections(
    detections: Dict[int, List[dict]],
    target: int,
    seg_start: int,
    seg_end: int,
    prefer: str = "before",      # "before" | "after"
) -> Optional[int]:
    frames = sorted(f for f in detections if seg_start <= f <= seg_end)
    if not frames:
        return None
    if prefer == "before":
        cands = [f for f in frames if f <= target]
        return cands[-1] if cands else min(frames, key=lambda f: abs(f - target))
    cands = [f for f in frames if f >= target]
    return cands[0] if cands else min(frames, key=lambda f: abs(f - target))


def greedy_new_cards(
    start_dets: List[dict],
    end_dets: List[dict],
    match_thresh: float = MATCH_DIST_THRESH,
) -> List[dict]:
    """
    Greedy nearest-neighbour matching between start and end detections.
    End-frame cards that cannot be matched to any start-frame card are 'new'.
    """
    if not start_dets:
        return list(end_dets)
    start_centres = [d["polygon_center"][0] for d in start_dets]
    dist_matrix = [
        [centre_dist(d["polygon_center"][0], sc) for sc in start_centres]
        for d in end_dets
    ]
    triples = sorted(
        [(i, j, dist_matrix[i][j])
         for i in range(len(end_dets))
         for j in range(len(start_dets))],
        key=lambda t: t[2],
    )
    matched_end, matched_start = set(), set()
    for i, j, dist in triples:
        if dist > match_thresh:
            break
        if i not in matched_end and j not in matched_start:
            matched_end.add(i)
            matched_start.add(j)
    return [end_dets[i] for i in range(len(end_dets)) if i not in matched_end]


def box_to_pct(box: List[int]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return (x1/IMG_W*100, y1/IMG_H*100,
            (x2-x1)/IMG_W*100, (y2-y1)/IMG_H*100)


def make_bounding_box_entry(frame: int, det: dict) -> dict:
    x_pct, y_pct, w_pct, h_pct = box_to_pct(det["box"])
    return {
        "frame": frame,
        "labels": ["card"],
        "keyframes": [{
            "frame": frame,
            "x": x_pct, "y": y_pct, "width": w_pct, "height": h_pct,
            "polygon_center": det["polygon_center"][0],
            "box": det["box"],
            "conf": det.get("conf"),
            "rank": det.get("rank", ""),
            "suit": det.get("suit", ""),
        }],
        "id": rand_id(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Filtering stages  (each returns (passed, failed) lists)
# ═══════════════════════════════════════════════════════════════════════════════

def stage1_duration(segments: List[dict], video_name_base: str) -> Tuple[List[dict], List[dict]]:
    """Keep only hit/dealer-hits segments with >= MIN_SEGMENT_FRAMES frames."""
    passed, failed = [], []
    for seg in segments:
        if seg["label"] not in HIT_LABELS:
            continue
        duration = seg["end_frame"] - seg["start_frame"] + 1
        if duration >= MIN_SEGMENT_FRAMES:
            passed.append(seg)
        else:
            failed.append({
                **_base(seg, video_name_base),
                "stage": "stage1_duration",
                "reason": f"duration={duration} < {MIN_SEGMENT_FRAMES}",
            })
    return passed, failed


def stage2_new_card(
    segments: List[dict],
    card_detections: Optional[Dict[int, List[dict]]],
    video_name_base: str,
) -> Tuple[List[dict], List[dict]]:
    """
    For each segment, find the newly appeared card by comparing start vs end
    frame detections.  Enriches passing segments with:
      seg["end_det_frame"], seg["new_cards"]
    Failing segments carry extra fields for visualisation:
      "start_dets", "end_dets"  (only for no_new_card_found)
    """
    if card_detections is None:
        failed = [{**_base(seg, video_name_base), "stage": "stage2_no_card_jsonl",
                   "reason": "card jsonl not found"}
                  for seg in segments]
        return [], failed

    passed, failed = [], []
    for seg in segments:
        sf, ef = seg["start_frame"], seg["end_frame"]

        start_det_f = nearest_frame_with_detections(
            card_detections, sf, sf, ef, prefer="after")
        end_det_f = nearest_frame_with_detections(
            card_detections, ef, sf, ef, prefer="before")

        if start_det_f is None or end_det_f is None:
            failed.append({**_base(seg, video_name_base), "stage": "stage2_no_detections",
                           "reason": "no card detections within segment"})
            continue

        if start_det_f >= end_det_f:
            failed.append({**_base(seg, video_name_base), "stage": "stage2_bad_frame_order",
                           "reason": f"start_det({start_det_f})>=end_det({end_det_f})"})
            continue

        start_dets = card_detections.get(start_det_f, [])
        end_dets   = card_detections.get(end_det_f,   [])
        new_cards  = greedy_new_cards(start_dets, end_dets)

        if not new_cards:
            failed.append({
                **_base(seg, video_name_base), "stage": "stage2_no_new_card",
                "reason": (f"start_f{start_det_f}:{len(start_dets)}dets"
                           f"->end_f{end_det_f}:{len(end_dets)}dets"),
                # kept for failure vis only – stripped before writing JSON
                "_start_dets": start_dets,
                "_end_dets": end_dets,
            })
            continue

        passed.append({**seg, "end_det_frame": end_det_f, "new_cards": new_cards})

    return passed, failed


def stage3_spatial(segments: List[dict], video_name_base: str) -> Tuple[List[dict], List[dict]]:
    """
    Apply per-label spatial constraint and pick one card per segment.
    Passing segments get a "chosen_card" key.
    Failing segments carry "_candidate_cards" for visualisation.
    """
    passed, failed = [], []
    for seg in segments:
        label     = seg["label"]
        new_cards = seg["new_cards"]

        if label == "dealer hits":
            (x1, y1), (x2, y2) = DEALER_BBOX_PX
            inside = [d for d in new_cards
                      if x1 <= d["polygon_center"][0][0] <= x2
                      and y1 <= d["polygon_center"][0][1] <= y2]
            if not inside:
                failed.append({
                    **_base(seg, video_name_base), "stage": "stage3_spatial",
                    "reason": (f"no centroid inside dealer bbox "
                               f"({len(new_cards)} candidates)"),
                    "_candidate_cards": new_cards,
                })
                continue
            chosen = max(inside, key=lambda d: d["polygon_center"][0][0])

        else:  # hit
            below = [d for d in new_cards
                     if d["polygon_center"][0][1] >= HIT_MIN_Y_PX]
            if not below:
                failed.append({
                    **_base(seg, video_name_base), "stage": "stage3_spatial",
                    "reason": (f"no centroid y>={HIT_MIN_Y_PX}px "
                               f"({len(new_cards)} candidates)"),
                    "_candidate_cards": new_cards,
                })
                continue
            chosen = max(below, key=lambda d: d["polygon_center"][0][1])

        passed.append({**seg, "chosen_card": chosen})

    return passed, failed


def _base(seg: dict, video_name_base: str) -> dict:
    """Minimal failure record fields shared across all stages."""
    return {
        "video": video_name_base,
        "label": seg["label"],
        "start_frame": seg["start_frame"],
        "end_frame": seg["end_frame"],
        "duration_frames": seg["end_frame"] - seg["start_frame"] + 1,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Visualisation
# ═══════════════════════════════════════════════════════════════════════════════

def _load_segment_frames(video_name_base: str, sf: int, ef: int):
    """
    Read frames [sf, ef] (inclusive, 0-indexed) from the raw filtered video.
    Returns (frames_list, fps), or ([], 25.0) on failure.
    """
    try:
        import cv2
    except ImportError:
        return [], 25.0
    # Raw videos: "YYYY-MM-DD HH-MM-SS.mp4"; video_name_base: "YYYY-MM-DD_HH-MM-SS_XXXXXX_YYYYYY"
    parts = video_name_base.split('_')
    base_video_name = f"{parts[0]}_{parts[1]}".replace('_', ' ', 1)
    clip_start_1idx = int(parts[2])
    video_path = os.path.join(RAW_VIDEO_DIR, f"{base_video_name}.mp4")
    if not os.path.exists(video_path):
        # some videos use underscore instead of space
        video_path = os.path.join(RAW_VIDEO_DIR, f"{parts[0]}_{parts[1]}.mp4")
    if not os.path.exists(video_path):
        return [], 25.0
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    # Seek to global frame: clip_start is 1-indexed, sf is relative to clip
    global_frame = (clip_start_1idx - 1) + sf
    cap.set(cv2.CAP_PROP_POS_FRAMES, global_frame)
    frames = []
    for _ in range(ef - sf + 1):
        ret, frame = cap.read()
        if not ret:
            break
        # resize to 720p to match spatial thresholds
        if frame.shape[1] != IMG_W or frame.shape[0] != IMG_H:
            frame = cv2.resize(frame, (IMG_W, IMG_H))
        frames.append(frame)
    cap.release()
    return frames, fps


def _write_video(frames, out_path: str, fps: float = 25.0):
    """Write frames as H.264 mp4 via a temp mp4v file re-encoded with ffmpeg."""
    import cv2
    import subprocess
    import tempfile
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if not frames:
        return
    h, w = frames[0].shape[:2]
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
    writer = cv2.VideoWriter(
        tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()
    subprocess.run(
        ["ffmpeg", "-y", "-i", tmp_path,
         "-vcodec", "libx264", "-crf", "23", "-preset", "fast",
         "-movflags", "+faststart", out_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        check=True,
    )
    os.remove(tmp_path)


def _draw_dets(img, dets: List[dict], color, prefix: str = ""):
    import cv2
    for d in dets:
        cx, cy = int(d["polygon_center"][0][0]), int(d["polygon_center"][0][1])
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.circle(img, (cx, cy), 6, color, -1)
        cv2.putText(img, f"{prefix}{d.get('rank','')}/{d.get('suit','')}",
                    (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def _draw_dealer_bbox(img):
    import cv2
    (x1, y1), (x2, y2) = DEALER_BBOX_PX
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 165, 255), 2)
    cv2.putText(img, "dealer bbox", (x1, max(0, y1-5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)


def _draw_hit_line(img):
    import cv2
    cv2.line(img, (0, HIT_MIN_Y_PX), (IMG_W, HIT_MIN_Y_PX), (0, 165, 255), 2)
    cv2.putText(img, f"y>={HIT_MIN_Y_PX}", (5, HIT_MIN_Y_PX - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)


def _overlay_failure(frame, stage: str, label: str, sf: int, ef: int,
                     reason: str, fc: dict):
    """Draw failure overlay onto a single frame in-place."""
    import cv2
    cv2.putText(frame, f"[{stage}] {label} f{sf}-{ef}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 165, 0), 2)
    cv2.putText(frame, reason,
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    if stage == "stage2_no_new_card":
        # blue = start-frame cards (already on table), green = end-frame cards
        _draw_dets(frame, fc.get("_start_dets", []), (255, 80, 0), "S:")
        _draw_dets(frame, fc.get("_end_dets",   []), (0, 220, 0),  "E:")
    elif stage == "stage3_spatial":
        _draw_dets(frame, fc.get("_candidate_cards", []), (0, 0, 255), "rej:")
        if label == "dealer hits":
            _draw_dealer_bbox(frame)
        else:
            _draw_hit_line(frame)


def visualize_failures(all_failures: List[dict], count: int):
    if count == 0:
        return
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("cv2 not available – skipping failure vis")
        return

    pool = list(all_failures)
    if count != -1 and count < len(pool):
        pool = random.sample(pool, count)

    total = len(pool)
    for idx, fc in enumerate(pool, 1):
        stage = fc["stage"]
        vname = fc["video"]
        label = fc["label"]
        sf, ef = fc["start_frame"], fc["end_frame"]
        clip_label = label.replace(" ", "_")
        print(f"  failure vis [{idx}/{total}] {vname} f{sf}-{ef} ({stage})", flush=True)
        out_path = str(FAILURE_VIS_DIR / stage / f"{clip_label}_{vname}_{sf}_{ef}.mp4")
        os.makedirs(str(FAILURE_VIS_DIR / stage), exist_ok=True)

        # ── stage1: extract the short segment from the raw video ──────────
        if stage == "stage1_duration":
            frames, fps = _load_segment_frames(vname, sf, ef)
            if frames:
                _write_video(frames, out_path, fps)
            else:
                with open(out_path.replace(".mp4", "_no_raw_video.txt"), "w") as fh:
                    fh.write(fc.get("reason", ""))
            continue

        # ── stage2 / stage3: overlay annotations on every frame ───────────
        frames, fps = _load_segment_frames(vname, sf, ef)
        reason = fc.get("reason", "")

        if not frames:
            assert False, f"No frames loaded for {vname} f{sf}-{ef} – raw video not found"

        for frame in frames:
            _overlay_failure(frame, stage, label, sf, ef, reason, fc)

        _write_video(frames, out_path, fps)

    print(f"Failure vis ({total} clips) saved under: {FAILURE_VIS_DIR}/")


def visualize_success(vis_segments: List[dict], count: int, out_dir: str):
    """
    Render up to `count` randomly sampled successful segments as H.264 clips
    with chosen-card bbox + constraint region overlaid on every frame.
    count=-1 visualises all.
    """
    if count == 0:
        return
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("cv2 not available – skipping success vis")
        return

    os.makedirs(out_dir, exist_ok=True)

    pool = list(vis_segments)
    if count != -1 and count < len(pool):
        pool = random.sample(pool, count)

    total = len(pool)
    for idx, info in enumerate(pool, 1):
        vname = info["video_name_base"]
        label = info["label"]
        sf, ef = info["start_frame"], info["end_frame"]
        print(f"  success vis [{idx}/{total}] {vname} f{sf}-{ef}", flush=True)

        frames, fps = _load_segment_frames(vname, sf, ef)
        assert frames, f"No frames loaded for {vname} f{sf}-{ef} – raw video not found"

        for frame in frames:
            _draw_dets(frame, [info["chosen_card"]], (0, 255, 0))
            if label == "dealer hits":
                _draw_dealer_bbox(frame)
            else:
                _draw_hit_line(frame)
            cv2.putText(frame, f"{label}  f{sf}-{ef}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)

        clip_label = label.replace(" ", "_")
        out_path = os.path.join(out_dir, f"{clip_label}_{vname}_{sf}_{ef}_vis.mp4")
        _write_video(frames, out_path, fps)

    print(f"Success vis ({total} clips) saved under: {out_dir}/")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def load_vis_segments_from_output(out_dir: Path) -> List[dict]:
    """Reconstruct vis-segment records from written annotation JSONs."""
    vis_segs = []
    for anno_file in sorted(out_dir.glob("*_annotations.json")):
        video_name_base = anno_file.stem.replace("_annotations", "")
        with open(anno_file) as f:
            record = json.load(f)
        for seg in record.get("timeline_segments", []):
            if not seg.get("bounding_boxes"):
                continue
            bb = seg["bounding_boxes"][0]
            kf = bb["keyframes"][0]
            # reconstruct the chosen_card dict from stored keyframe fields
            chosen_card = {
                "polygon_center": [kf["polygon_center"]],
                "box": kf["box"],
                "conf": kf.get("conf"),
                "rank": kf.get("rank", ""),
                "suit": kf.get("suit", ""),
            }
            vis_segs.append({
                "video_name_base": video_name_base,
                "label": seg["labels"][0],
                "start_frame": seg["start_frame"],
                "end_frame": seg["end_frame"],
                "end_det_frame": kf["frame"],
                "chosen_card": chosen_card,
                "bounding_boxes": seg["bounding_boxes"],
            })
    return vis_segs


def _write_annotation(video_name_base: str, video_name: str,
                      timeline_segments: List[dict]):
    """Write a single video's annotation JSON immediately (supports resume)."""
    record = {
        "video_name": video_name,
        "video_id": "",
        "video_path": "",
        "timeline_segments": timeline_segments,
    }
    out_path = OUTPUT_DIR / f"{video_name_base}_annotations.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)


def process(vis_count: int = 0, fail_vis_count: int = 0, max_videos: int = 0) -> List[dict]:
    with open(SEGMENTS_JSON) as f:
        all_segments: Dict[str, List[dict]] = json.load(f)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── detect already-processed videos for resume ────────────────────────
    existing = {p.stem.replace("_annotations", "")
                for p in OUTPUT_DIR.glob("*_annotations.json")}
    if existing:
        print(f"Resuming: {len(existing)} videos already processed, skipping them.")

    all_failures: List[dict] = []
    vis_segments: List[dict] = []
    video_keys = list(all_segments.keys())
    if max_videos > 0:
        video_keys = video_keys[:max_videos]
    total_videos = len(video_keys)
    n_skipped = 0
    n_processed = 0

    for vi, txt_key in enumerate(video_keys, 1):
        segments = all_segments[txt_key]
        video_name_base = txt_key.replace(".txt", "")
        video_name = video_name_base + ".mp4"

        # ── resume: skip already-written annotation JSONs ─────────────────
        if video_name_base in existing:
            n_skipped += 1
            continue

        # ── Stage 1: duration ─────────────────────────────────────────────
        s1_pass, s1_fail = stage1_duration(segments, video_name_base)
        all_failures.extend(s1_fail)

        if not s1_pass:
            # still write an empty annotation so resume skips this video
            _write_annotation(video_name_base, video_name, [])
            n_processed += 1
            if n_processed % 50 == 0:
                print(f"[{vi}/{total_videos}] {n_processed} processed, "
                      f"{n_skipped} skipped …", flush=True)
            continue

        # load card detections once per video
        card_jsonl = os.path.join(CARDS_DIR, f"{video_name_base}_card.jsonl")
        card_dets = load_card_detections(card_jsonl) if os.path.exists(card_jsonl) else None

        print(f"[{vi}/{total_videos}] {video_name_base}  "
              f"({len(s1_pass)} hit/dealer-hits segs) …", flush=True)

        # ── Stage 2: new card diff ────────────────────────────────────────
        s2_pass, s2_fail = stage2_new_card(s1_pass, card_dets, video_name_base)
        all_failures.extend(s2_fail)

        # ── Stage 3: spatial filter ───────────────────────────────────────
        s3_pass, s3_fail = stage3_spatial(s2_pass, video_name_base)
        all_failures.extend(s3_fail)

        # ── Build output timeline (hit / dealer hits only) ────────────────
        hit_success = {(s["start_frame"], s["end_frame"]): s for s in s3_pass}

        timeline_segments = []
        for seg in segments:
            if seg["label"] not in HIT_LABELS:
                continue
            sf, ef = seg["start_frame"], seg["end_frame"]
            duration = ef - sf + 1
            if duration < MIN_SEGMENT_FRAMES:
                continue
            bboxes = []
            if (sf, ef) in hit_success:
                chosen = hit_success[(sf, ef)]["chosen_card"]
                end_det_f = hit_success[(sf, ef)]["end_det_frame"]
                bboxes = [make_bounding_box_entry(end_det_f, chosen)]
            timeline_segments.append({
                "start_frame": sf, "end_frame": ef,
                "labels": [seg["label"]],
                "meta_text": [],
                "duration_frames": duration,
                "id": rand_id(),
                "bounding_boxes": bboxes,
            })

        # write immediately so interrupted runs can resume
        _write_annotation(video_name_base, video_name, timeline_segments)

        # collect for success visualisation (needs chosen_card)
        for s in s3_pass:
            vis_segments.append({
                "video_name_base": video_name_base,
                "label": s["label"],
                "start_frame": s["start_frame"],
                "end_frame": s["end_frame"],
                "end_det_frame": s["end_det_frame"],
                "chosen_card": s["chosen_card"],
                "bounding_boxes": [make_bounding_box_entry(
                    s["end_det_frame"], s["chosen_card"])],
            })

        n_processed += 1

    # ── Stats ─────────────────────────────────────────────────────────────────
    stage_counts: Dict[str, int] = {}
    for fc in all_failures:
        stage_counts[fc["stage"]] = stage_counts.get(fc["stage"], 0) + 1

    total_hit = len(vis_segments) + len(all_failures)
    stats = {
        "total_hit_dealer_hits_segments": total_hit,
        "successful": len(vis_segments),
        "failed_total": len(all_failures),
        "failed_by_stage": stage_counts,
        "skipped_resume": n_skipped,
    }
    with open(OUTPUT_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # strip internal _-prefixed keys before writing failure JSON
    def _clean(fc: dict) -> dict:
        return {k: v for k, v in fc.items() if not k.startswith("_")}

    with open(OUTPUT_DIR / "failure_cases.json", "w") as f:
        json.dump([_clean(fc) for fc in all_failures], f, indent=2)

    print(f"\nSummary: {len(vis_segments)} successful  /  "
          f"{len(all_failures)} failed  /  {total_hit} total"
          f"  ({n_skipped} skipped from previous run)")
    for stage, cnt in sorted(stage_counts.items()):
        print(f"  {stage}: {cnt}")
    print(f"Stats   : {OUTPUT_DIR / 'stats.json'}")
    print(f"Failures: {OUTPUT_DIR / 'failure_cases.json'}")

    # ── Failure visualisation (optional) ──────────────────────────────────────
    if fail_vis_count != 0:
        n = len(all_failures) if fail_vis_count == -1 else fail_vis_count
        print(f"\nGenerating {n} failure visualisations …")
        visualize_failures(all_failures, fail_vis_count)

    # ── Success visualisation (optional) ──────────────────────────────────────
    if vis_count != 0:
        n = len(vis_segments) if vis_count == -1 else vis_count
        print(f"\nGenerating {n} success visualisations …")
        visualize_success(vis_segments, vis_count, str(SUCCESS_VIS_DIR))

    return vis_segments


# ─── entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vis_count", type=int, default=0,
        help="Success clips to render: 0=none (default), N=random N, -1=all",
    )
    parser.add_argument(
        "--fail_vis_count", type=int, default=0,
        help="Failure clips to render: 0=none (default), N=random N, -1=all",
    )
    parser.add_argument(
        "--vis_only", action="store_true",
        help=(
            "Skip filtering; reconstruct vis segments from written annotation JSONs "
            "and only generate success visualisations."
        ),
    )
    parser.add_argument(
        "--max_videos", type=int, default=0,
        help="Limit processing to first N videos (0=all, default)",
    )
    args = parser.parse_args()

    if args.vis_only:
        vis_segs = load_vis_segments_from_output(OUTPUT_DIR)
        n = len(vis_segs) if args.vis_count == -1 else args.vis_count
        print(f"Loaded {len(vis_segs)} segments from annotation JSONs. Visualising {n} …")
        visualize_success(vis_segs, args.vis_count, str(SUCCESS_VIS_DIR))
        if args.fail_vis_count != 0:
            fc_path = OUTPUT_DIR / "failure_cases.json"
            if fc_path.exists():
                with open(fc_path) as f:
                    all_failures = json.load(f)
                fn = len(all_failures) if args.fail_vis_count == -1 else args.fail_vis_count
                print(f"Loaded {len(all_failures)} failure cases. Visualising {fn} …")
                visualize_failures(all_failures, args.fail_vis_count)
            else:
                print(f"No failure_cases.json found at {fc_path}")
    else:
        process(vis_count=args.vis_count, fail_vis_count=args.fail_vis_count, max_videos=args.max_videos)
