"""Demo: visualize predict_keypoints() output on N sample videos.

Picks evenly-spaced videos from SRC_DIR, runs predict_keypoints on the
middle frame of each, draws scanner + shoe keypoints, writes annotated
PNGs to vis_predict/ next to this script.
"""
import argparse
from pathlib import Path

import cv2

from predict_keypoints import load_model, predict_keypoints

SRC_DIR = Path('/home/ubuntu/us-west-3-fs/somrita/livedealer/roundcut/batch_01')
OUT_DIR = Path(__file__).resolve().parent / 'vis_predict'

CLASS_COLORS = {
    'scanner': (0,   0, 255),   # red
    'shoe':    (0, 255,   0),   # green
}


def middle_frame(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def draw(frame, kpts: dict):
    for cname, points in kpts.items():
        if not points:
            continue
        color = CLASS_COLORS.get(cname, (255, 255, 255))
        for kid, (x, y, c) in points.items():
            cx, cy = int(x), int(y)
            cv2.circle(frame, (cx, cy), 5, color, -1)
            cv2.circle(frame, (cx, cy), 6, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, f'{cname[0]}{kid}', (cx + 6, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return frame


def pick_videos(src: Path, n: int):
    vids = sorted(src.glob('*.mp4'))
    if len(vids) <= n:
        return vids
    step = (len(vids) - 1) / (n - 1)
    return [vids[round(i * step)] for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--src-dir', default=str(SRC_DIR))
    ap.add_argument('--out-dir', default=str(OUT_DIR))
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    vids = pick_videos(Path(args.src_dir), args.n)
    print(f'visualizing {len(vids)} videos -> {out}')

    model = load_model()
    for i, v in enumerate(vids, 1):
        frame = middle_frame(v)
        if frame is None:
            print(f'[{i}/{len(vids)}] {v.name}  FAIL')
            continue
        kpts = predict_keypoints(model, frame)
        annotated = draw(frame, kpts)
        cv2.imwrite(str(out / (v.stem + '.png')), annotated)
        n_scan = len(kpts['scanner'] or {})
        n_shoe = len(kpts['shoe'] or {})
        print(f'[{i}/{len(vids)}] {v.name}  scanner_kpts={n_scan}  shoe_kpts={n_shoe}')


if __name__ == '__main__':
    main()
