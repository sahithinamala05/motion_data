"""Render H.264 overlay videos for a sample of roundcut videos using
per-video JSONs in shoe_scanner/json/. Boxes are static (1st frame) so
the overlay is identical for every frame.

Uses an ffmpeg overlay filter with a transparent PNG so we only decode +
re-encode the video stream once, in libx264.
"""
import argparse
import json
import random
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

JSON_DIR = Path('/home/ubuntu/us-west-3-fs/sahithi/shoe_scanner/json')
SRC_DIR  = Path('/home/ubuntu/us-west-3-fs/live_dealer_blackjack/roundcut/good_quality_rounds')
OUT_DIR  = Path('/home/ubuntu/us-west-3-fs/sahithi/shoe_scanner/videos')
FFMPEG   = '/usr/bin/ffmpeg'

# BGRA per class (alpha=255 fully opaque)
COLORS = {
    'discard_tray': (0, 215, 255, 255),   # gold
    'shoe':         (0, 255,   0, 255),   # green
    'spare_shoe':   (255, 128,  0, 255),  # blue-cyan
    'scanner':      (0,   0, 255, 255),   # red
}


def build_overlay_png(width, height, detections, out_png):
    """Build a transparent overlay PNG with boxes + labels."""
    canvas = np.zeros((height, width, 4), dtype=np.uint8)  # BGRA
    for d in detections:
        x1, y1, x2, y2 = map(int, d['bbox_xyxy'])
        color = COLORS.get(d['class'], (255, 255, 255, 255))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{d['class']} {d['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ytxt = max(th + 4, y1)
        cv2.rectangle(canvas, (x1, ytxt - th - 4), (x1 + tw + 4, ytxt), color, -1)
        cv2.putText(canvas, label, (x1 + 2, ytxt - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_png), canvas)


def render_one(src_video: Path, overlay_png: Path, out_mp4: Path, crf=23):
    """Use ffmpeg to overlay the static PNG onto the video and encode H.264."""
    cmd = [
        FFMPEG, '-y', '-loglevel', 'error',
        '-i', str(src_video),
        '-i', str(overlay_png),
        '-filter_complex', '[0:v][1:v]overlay=0:0[v]',
        '-map', '[v]',
        '-c:v', 'libx264', '-preset', 'veryfast', '-crf', str(crf),
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        '-an',
        str(out_mp4),
    ]
    return subprocess.run(cmd, capture_output=True).returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--json-dir', default=str(JSON_DIR))
    ap.add_argument('--src-dir',  default=str(SRC_DIR))
    ap.add_argument('--out-dir',  default=str(OUT_DIR))
    ap.add_argument('--crf', type=int, default=23)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jsons = sorted(Path(args.json_dir).glob('*.json'))
    if not jsons:
        raise SystemExit(f'No JSONs in {args.json_dir} — run inference first.')

    rng = random.Random(args.seed)
    sample = rng.sample(jsons, min(args.n, len(jsons)))

    n_ok = n_fail = n_skip = 0
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for i, j in enumerate(sample, 1):
            rec = json.loads(j.read_text())
            src = Path(args.src_dir) / rec['video']
            out = out_dir / (Path(rec['video']).stem + '.mp4')
            if not src.exists():
                n_skip += 1
                continue
            if out.exists():
                n_ok += 1
                continue
            overlay = td / (j.stem + '.png')
            build_overlay_png(rec['width'], rec['height'], rec['detections'], overlay)
            if render_one(src, overlay, out, crf=args.crf):
                n_ok += 1
            else:
                n_fail += 1
            if i % 25 == 0 or i == len(sample):
                print(f'  {i:4d}/{len(sample)}  ok={n_ok}  fail={n_fail}  skip={n_skip}',
                      flush=True)
    print(f'Done. ok={n_ok}  fail={n_fail}  skip={n_skip}  out={out_dir}')


if __name__ == '__main__':
    main()
