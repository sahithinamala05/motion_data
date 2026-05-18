"""
Process instance_segments.json to extract verified clean hand segments.

Pipeline:
  OCR filter : reject segments with "Game paused due to inactivity" text.

Output: annotation JSON per video with clean hand timeline segments.

Usage:
    python process_clean_hand_segments.py [--vis_count N] [--fail_vis_count N]
                                          [--max_videos N] [--workers N]
"""

import argparse
import json
import multiprocessing as mp
import os
import random
import re
import string
import subprocess
import tempfile
from pathlib import Path

import cv2
import easyocr

# ─── Paths ───────────────────────────────────────────────────────────────────
SEGMENTS_JSON = (
    "/home/ubuntu/us-west-3-fs/sahithi/hard_action_annotations/"
    "fact_preds_sparse245163_iter26000_cleanhandsplit_allvideos/instance_segments.json"
)
RAW_VIDEO_DIR = "/home/ubuntu/us-west-3-fs/live_dealer_blackjack/raw/batch_01"
OUTPUT_DIR = Path(
    "/home/ubuntu/us-west-3-fs/sahithi/hard_action_annotations/full_run/clean_hand"
)
SUCCESS_VIS_DIR = OUTPUT_DIR / "success_vis"
FAILURE_VIS_DIR = OUTPUT_DIR / "failure_vis"

# ─── Constants ───────────────────────────────────────────────────────────────
IMG_W, IMG_H = 1280, 720
CLEAN_HAND_LABEL = "clean hand"
# Permissive regexes for each keyword to tolerate interior OCR misreads
# (e.g. EasyOCR reading "INACTIVITY" as "INACTWVITY").
PAUSE_KEYWORD_RES = (
    re.compile(r"\bpaus\w*d\b", re.IGNORECASE),         # paused / paus3d / etc.
    re.compile(r"\binact\w{2,6}ity\b", re.IGNORECASE),  # inactivity / inactwvity / etc.
)

# Failure reason codes
REASON_OK = "ok"
REASON_PAUSE = "paused due to inactivity text detected"
REASON_NO_FRAMES = "could not load frames"


# ─── Utilities ───────────────────────────────────────────────────────────────

def rand_id(n=10):
    return "".join(random.choices(string.ascii_letters + string.digits, k=n))


def _get_video_path(vname):
    """Map `vname` → (video_path, clip_start_frame). Returns (None, None) if missing."""
    parts = vname.split("_")
    base = f"{parts[0]}_{parts[1]}".replace("_", " ", 1)
    for name in (f"{base}.mp4", f"{parts[0]}_{parts[1]}.mp4"):
        path = os.path.join(RAW_VIDEO_DIR, name)
        if os.path.exists(path):
            return path, int(parts[2])
    return None, None


def _resize_if_needed(frame):
    if frame.shape[1] != IMG_W or frame.shape[0] != IMG_H:
        frame = cv2.resize(frame, (IMG_W, IMG_H))
    return frame


def _load_all_frames(vname, sf, ef):
    """Load every frame in [sf, ef]. Returns (frames, fps)."""
    path, clip_start = _get_video_path(vname)
    if path is None:
        return [], 25.0

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, (clip_start - 1) + sf)

    frames = []
    for _ in range(ef - sf + 1):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(_resize_if_needed(frame))
    cap.release()
    return frames, fps


# ─── OCR check ───────────────────────────────────────────────────────────────

_reader = None


def _get_reader():
    """Lazy-init a single EasyOCR reader on GPU (per process)."""
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    return _reader


def _has_pause_text(frame):
    """Run EasyOCR on the frame; return True if ANY pause keyword is detected."""
    try:
        texts = _get_reader().readtext(frame, detail=0, paragraph=False)
    except Exception:
        return False
    joined = " ".join(texts)
    return any(rx.search(joined) for rx in PAUSE_KEYWORD_RES)


# ─── Per-video worker (called in pool workers) ───────────────────────────────

def _init_worker():
    """Pool initializer: warm up the EasyOCR reader in each child process."""
    _get_reader()


def _process_video(item):
    """Process all clean-hand segments for one video.

    Opens the video file ONCE, samples + OCRs each segment, writes the
    annotation JSON. Returns (vname, pass_list, fail_list, counters).
    """
    vname, segments = item

    pass_list, fail_list = [], []
    counters = {"total": 0, "passed": 0,
                "dropped_pause": 0, "dropped_noframes": 0}
    timeline = []

    path, clip_start = _get_video_path(vname)
    cap = cv2.VideoCapture(path) if path else None

    for seg in segments:
        sf, ef = seg["start_frame"], seg["end_frame"]
        counters["total"] += 1

        # Scan every frame in the segment; early-exit on first keyword hit.
        any_frame_read = False
        pause_hit = False
        if cap is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, (clip_start - 1) + sf)
            for _ in range(ef - sf + 1):
                ret, frame = cap.read()
                if not ret:
                    break
                any_frame_read = True
                if _has_pause_text(_resize_if_needed(frame)):
                    pause_hit = True
                    break

        if not any_frame_read:
            counters["dropped_noframes"] += 1
            fail_list.append((vname, seg, REASON_NO_FRAMES))
            continue

        if pause_hit:
            counters["dropped_pause"] += 1
            fail_list.append((vname, seg, REASON_PAUSE))
            continue

        timeline.append({
            "start_frame": sf,
            "end_frame": ef,
            "labels": [CLEAN_HAND_LABEL],
            "meta_text": [],
            "duration_frames": ef - sf + 1,
            "id": rand_id(),
            "bounding_boxes": [],
        })
        counters["passed"] += 1
        pass_list.append((vname, seg, REASON_OK))

    if cap is not None:
        cap.release()

    _write_annotation(vname, timeline)
    return vname, pass_list, fail_list, counters


# ─── Annotation + visualization ──────────────────────────────────────────────

def _write_annotation(vname, timeline):
    out_path = OUTPUT_DIR / f"{vname}_annotations.json"
    with open(out_path, "w") as f:
        json.dump({
            "video_name": f"{vname}.mp4",
            "video_id": "",
            "video_path": "",
            "timeline_segments": timeline,
        }, f, indent=2)


def _write_video(frames, out_path, fps):
    if not frames:
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    h, w = frames[0].shape[:2]

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()

    subprocess.run(
        ["/usr/bin/ffmpeg", "-y", "-i", tmp_path, "-vcodec", "libx264",
         "-crf", "23", "-preset", "fast", "-movflags", "+faststart", out_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
    )
    os.remove(tmp_path)


def visualize_segments(segments, count, out_dir, label, color):
    """segments: list of (vname, seg, reason). count: 0=none, N=random N, -1=all."""
    os.makedirs(out_dir, exist_ok=True)
    pool = list(segments)
    if count != -1 and count < len(pool):
        pool = random.sample(pool, count)

    for idx, (vname, seg, reason) in enumerate(pool, 1):
        sf, ef = seg["start_frame"], seg["end_frame"]
        print(f"  {label} vis [{idx}/{len(pool)}] {vname} f{sf}-{ef}", flush=True)

        frames, fps = _load_all_frames(vname, sf, ef)
        if not frames:
            continue

        dur = ef - sf + 1
        for i, frame in enumerate(frames):
            cv2.putText(frame, f"{label} CLEAN HAND f{sf+i} (seg {sf}-{ef}, {dur}f)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if reason:
                cv2.putText(frame, reason[:100], (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        out_path = os.path.join(out_dir, f"cleanhand_{vname}_{sf}_{ef}.mp4")
        _write_video(frames, out_path, fps)

    print(f"{label} vis ({len(pool)} clips) saved under: {out_dir}/")


# ─── Main pipeline ───────────────────────────────────────────────────────────

def _load_clean_hand_videos():
    with open(SEGMENTS_JSON) as f:
        all_segments = json.load(f)

    return {
        vname: [s for s in segs if s["label"] == CLEAN_HAND_LABEL]
        for vname, segs in all_segments.items()
        if any(s["label"] == CLEAN_HAND_LABEL for s in segs)
    }


def process(vis_count=0, fail_vis_count=0, max_videos=0, workers=4):
    print(f"Loading instance segments from: {SEGMENTS_JSON}", flush=True)
    clean_hand_videos = _load_clean_hand_videos()
    n_ch = sum(len(v) for v in clean_hand_videos.values())
    print(f"Found {n_ch} clean hand segments across {len(clean_hand_videos)} videos",
          flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    work = list(clean_hand_videos.items())
    if max_videos:
        work = work[:max_videos]
    n_videos = len(work)

    print(f"Running with {workers} workers …", flush=True)

    total = passed = dropped_pause = dropped_noframes = 0
    pass_pool, fail_pool = [], []

    with mp.Pool(workers, initializer=_init_worker) as pool:
        for vi, (vname, pl, fl, c) in enumerate(
                pool.imap_unordered(_process_video, work, chunksize=4), 1):
            pass_pool.extend(pl)
            fail_pool.extend(fl)
            total += c["total"]
            passed += c["passed"]
            dropped_pause += c["dropped_pause"]
            dropped_noframes += c["dropped_noframes"]

            if vi % 200 == 0 or vi == n_videos:
                print(f"[{vi}/{n_videos}] passed={passed} "
                      f"pause={dropped_pause} noframes={dropped_noframes}",
                      flush=True)

    # Stats
    stats = {
        "total": total,
        "passed": passed,
        "dropped_pause": dropped_pause,
        "dropped_noframes": dropped_noframes,
    }
    with open(OUTPUT_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSummary: {passed} passed / {total} total")
    print(f"  dropped_pause (OCR match): {dropped_pause}")
    print(f"  dropped_noframes: {dropped_noframes}")

    if vis_count != 0:
        print("\nGenerating success vis …")
        visualize_segments(pass_pool, vis_count, str(SUCCESS_VIS_DIR),
                           label="PASS", color=(0, 255, 0))

    if fail_vis_count != 0:
        print("\nGenerating failure vis …")
        visualize_segments(fail_pool, fail_vis_count, str(FAILURE_VIS_DIR),
                           label="FAIL", color=(0, 0, 255))


if __name__ == "__main__":
    # CUDA requires spawn start method when forking from parent
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vis_count", type=int, default=0,
                        help="Success clips: 0=none, N=random N, -1=all")
    parser.add_argument("--fail_vis_count", type=int, default=0,
                        help="Failure clips: 0=none, N=random N, -1=all")
    parser.add_argument("--max_videos", type=int, default=0,
                        help="Limit to first N videos (0=all)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel worker processes (default 4)")
    args = parser.parse_args()
    process(
        vis_count=args.vis_count,
        fail_vis_count=args.fail_vis_count,
        max_videos=args.max_videos,
        workers=args.workers,
    )
