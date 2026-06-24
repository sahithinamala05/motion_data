"""
Visualize card adjustment during initial hands (1st + 2nd) + discard segments
on one round, using roundcut card detection JSONLs.

Overlays per frame:
  - bbox + rank/suit label for each detected card
  - fading trail of each (rank, suit)'s polygon_center across recent frames
  - banner: current segment label + frame index
"""
import argparse
import os
import sys
import json
import subprocess
from collections import defaultdict, deque

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "10s-chunk"))
from clustering_util import assign_cards_to_positions  # noqa: E402

DEFAULT_PRED_DIR  = "/home/ubuntu/us-west-3-fs/sahithi/Ouput/remaining/predictions_json"
DEFAULT_CARDS_DIR = "/home/ubuntu/us-west-3-fs/sahithi/hard_action_annotations/card_detections_roundwise"
DEFAULT_VIDEO_DIR = "/home/ubuntu/us-west-3-fs/live_dealer_blackjack/roundcut/remaining_vid"
DEFAULT_OUTPUT_DIR = "/home/ubuntu/us-west-3-fs/sahithi/Ouput/viz_initial_hands"

SCALE = 1280 / 1920   # card JSONL coords are 1920x1080; video is 1280x720
LABELS_TO_RENDER = {"initial hands 2nd"}
TRAIL_LEN = 60        # frames of trail history per (rank, suit)

# Adjustment detection: per (rank, suit) trajectory inside the player zone, find
# the STABLE center — the mean position of the LONGEST run of consecutive
# detections that stayed within STABLE_TOL_PX of one another. Compare to the
# final (last detected) center. If their euclidean distance is >=
# ADJUST_THRESH_PX, the card was adjusted; the adjustment frame is the first
# frame after that stable run where the position diverges by >= ADJUST_THRESH_PX.
ADJUST_THRESH_PX = 8        # < 8 px between stable and final → no adjustment
STABLE_TOL_PX    = 5        # max pairwise spread inside a stable run
STABLE_MIN_LEN   = 10       # min consecutive frames to qualify as stable
PAD_SECONDS      = 5.0
PLAYER_MIN_Y_PX  = 480      # boundary line; only consider cards with y > this (below line, player zone)


def load_card_jsonl(path):
    by_frame = defaultdict(list)
    with open(path) as f:
        for line in f:
            o = json.loads(line)
            if o.get("detections"):
                by_frame[o["frame_id"]] = o["detections"]
    return by_frame


def scale_det(det):
    pc = det["polygon_center"]
    cx, cy = pc[0] if isinstance(pc[0], list) else pc
    x1, y1, x2, y2 = det["box"]
    return {
        **det,
        "center_px": (int(cx * SCALE), int(cy * SCALE)),
        "box_px": (int(x1 * SCALE), int(y1 * SCALE), int(x2 * SCALE), int(y2 * SCALE)),
    }


def cluster_by_seat(raw_dets):
    """Flatten polygon_center for clustering and return {seat_id: [card_det]}.
    Detections in returned dict have polygon_center as flat [x, y] (1920x1080)."""
    flat = []
    for d in raw_dets:
        pc = d.get("polygon_center")
        if isinstance(pc, list) and pc and isinstance(pc[0], list):
            flat.append({**d, "polygon_center": pc[0]})
        else:
            flat.append(d)
    return assign_cards_to_positions(flat)


def card_key(det):
    r = det.get("rank") or "?"
    s = det.get("suit") or "?"
    return f"{r}{s}"


def color_for_key(key, _cache={}):
    if key not in _cache:
        # deterministic pseudo-random color from hash of key
        h = abs(hash(key))
        _cache[key] = ((h >> 0) & 255, (h >> 8) & 255, (h >> 16) & 255)
    return _cache[key]


def _dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _longest_stable_run(pts):
    """Return (mean_xy, start_idx, end_idx) of the FIRST run of consecutive
    detections whose pairwise spread (max - min) on both axes is <= STABLE_TOL_PX
    and length >= STABLE_MIN_LEN. The earliest qualifying run is returned so
    "stable" anchors to where the card first settled, not where it ended up.
    None if no qualifying run exists."""
    n = len(pts)
    i = 0
    while i < n:
        j = i + 1
        xmin = xmax = pts[i][1][0]
        ymin = ymax = pts[i][1][1]
        while j < n:
            x, y = pts[j][1]
            nxmin = min(xmin, x); nxmax = max(xmax, x)
            nymin = min(ymin, y); nymax = max(ymax, y)
            if (nxmax - nxmin) > STABLE_TOL_PX or (nymax - nymin) > STABLE_TOL_PX:
                break
            xmin, xmax, ymin, ymax = nxmin, nxmax, nymin, nymax
            j += 1
        length = j - i
        if length >= STABLE_MIN_LEN:
            xs = [p[1][0] for p in pts[i:j]]
            ys = [p[1][1] for p in pts[i:j]]
            return (sum(xs) / length, sum(ys) / length), i, j - 1
        i = max(i + 1, j)
    return None


def _spatial_cluster(positions, k, max_iter=25):
    """2D k-means clustering. Returns a label (0..k-1) per point. Deterministic
    init via farthest-point seeding (no randomness)."""
    n = len(positions)
    if n == 0 or k <= 1:
        return [0] * n
    if k >= n:
        return list(range(n)) + [0] * (n - k if n > k else 0)

    # farthest-point seeding
    centroids = [positions[0]]
    for _ in range(k - 1):
        best = None
        best_d = -1.0
        for p in positions:
            d_min = min((p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 for c in centroids)
            if d_min > best_d:
                best_d = d_min
                best = p
        centroids.append(best)

    labels = [0] * n
    for _ in range(max_iter):
        changed = False
        new_labels = [0] * n
        for i, p in enumerate(positions):
            best_c = 0
            best_d = (p[0] - centroids[0][0]) ** 2 + (p[1] - centroids[0][1]) ** 2
            for j in range(1, k):
                d = (p[0] - centroids[j][0]) ** 2 + (p[1] - centroids[j][1]) ** 2
                if d < best_d:
                    best_d = d
                    best_c = j
            new_labels[i] = best_c
            if best_c != labels[i]:
                changed = True
        labels = new_labels
        if not changed:
            break
        # recompute centroids
        new_centroids = []
        for j in range(k):
            members = [positions[i] for i in range(n) if labels[i] == j]
            if members:
                new_centroids.append((
                    sum(p[0] for p in members) / len(members),
                    sum(p[1] for p in members) / len(members),
                ))
            else:
                new_centroids.append(centroids[j])
        centroids = new_centroids
    return labels


def _containing_segment(frame, segments):
    for s in segments:
        if s["start_frame"] <= frame <= s["end_frame"]:
            return s
    return None


def detect_adjustment_clips(cards_by_frame, frame_lo, frame_hi, fps, segments=None):
    """Detect adjustments INDEPENDENTLY inside each segment. A card is "adjusted"
    if, restricted to a single segment's frames, its stable_center (longest
    stationary run within the segment) differs from its final_center (last
    detection within the segment) by >= ADJUST_THRESH_PX. Cross-segment
    trajectories are never compared.

    Returns (merged_clips, events).
    """
    if segments is None:
        segments = [{"start_frame": frame_lo, "end_frame": frame_hi, "labels": [""]}]

    events = []
    for seg in segments:
        s_lo, s_hi = seg["start_frame"], seg["end_frame"]

        # Pass 1: per seat, gather (frame, center_px, rank+suit) for every detection.
        # Record max simultaneous cards per seat (K) so we know how many spatial
        # clusters to split into.
        per_seat = defaultdict(list)   # seat -> [(f, (x,y), rs)]
        max_K     = defaultdict(int)   # seat -> max cards per frame
        for f in range(s_lo, s_hi + 1):
            clusters = cluster_by_seat(cards_by_frame.get(f, []))
            for seat in range(1, 8):
                row = []
                for det in clusters.get(seat, []):
                    sd = scale_det(det)
                    if sd["center_px"][1] <= PLAYER_MIN_Y_PX:
                        continue
                    rs = card_key(sd)        # keep "?" / "U" labels so positions
                    row.append((f, sd["center_px"], rs))   # still contribute to clusters
                per_seat[seat].extend(row)
                if len(row) > max_K[seat]:
                    max_K[seat] = len(row)

        # Pass 2: per seat, drop infrequent-label outliers (cards labeled in
        # <5% of frames are usually detector mistakes), then spatially cluster
        # the remaining detections into K groups, where K is the max simultaneous
        # detection count restricted to the dominant labels. Each cluster gets
        # its majority-vote rank+suit as canonical label.
        from collections import Counter
        traj = {}
        for seat, all_dets in per_seat.items():
            if not all_dets:
                continue
            label_counts = Counter(d[2] for d in all_dets)
            total = sum(label_counts.values())
            dominant = {l for l, c in label_counts.items()
                        if c >= max(5, total * 0.05)
                        and not (l.startswith("?") or l.endswith("?")
                                 or l.startswith("U") or l.endswith("U"))}
            dets = [d for d in all_dets if d[2] in dominant]
            if not dets:
                continue
            # recompute max per-frame from dominant detections
            per_frame_n = Counter(d[0] for d in dets)
            K = max(1, max(per_frame_n.values()))
            assignments = _spatial_cluster([d[1] for d in dets], K)
            grouped = defaultdict(list)
            for (f, pos, rs), cid in zip(dets, assignments):
                grouped[cid].append((f, pos, rs))
            # Sort cluster IDs by mean y (smaller y = upper card)
            cids = sorted(grouped.keys(), key=lambda c: sum(p[1] for _, p, _ in grouped[c]) / len(grouped[c]))
            for sort_idx, cid in enumerate(cids):
                rows = grouped[cid]
                valid = [rs for _, _, rs in rows
                         if not (rs.startswith("?") or rs.endswith("?")
                                 or rs.startswith("U") or rs.endswith("U"))]
                canonical = Counter(valid).most_common(1)[0][0] if valid else "??"
                if K == 1:
                    suffix = ""
                elif K == 2:
                    suffix = "_U" if sort_idx == 0 else "_D"
                else:
                    suffix = f"_p{sort_idx}"   # top-to-bottom index for K>=3
                key = f"{canonical}@P{seat}{suffix}"
                traj[key] = [(f, pos) for f, pos, _ in rows]

        for key, pts in traj.items():
            if len(pts) < 2 * STABLE_MIN_LEN:
                continue
            res = _longest_stable_run(pts)
            if res is None:
                continue
            stable_xy, stable_start_idx, stable_end_idx = res
            stable_frame = pts[stable_end_idx][0]
            # require a SECOND stable run AFTER the first one (temporal step)
            tail = pts[stable_end_idx + 1:]
            res2 = _longest_stable_run(tail)
            if res2 is None:
                continue   # never re-settled → just settling jitter or noise
            final_xy, final_start_idx, final_end_idx = res2
            final_frame = tail[final_end_idx][0]
            delta = _dist(stable_xy, final_xy)
            if delta < ADJUST_THRESH_PX:
                continue
            adj_frame = next(
                (pts[i][0] for i in range(stable_end_idx + 1, len(pts))
                 if _dist(pts[i][1], stable_xy) >= ADJUST_THRESH_PX),
                final_frame,
            )
            # Parse "<rank+suit>@P<seat>{_U/_D/_pN}" → identifying parts
            head, _, suff = key.partition("@")
            seat_str, _, pos_str = suff.partition("_")
            player = seat_str.lstrip("P")
            position_label = pos_str if pos_str else "single"
            events.append({
                "key": key,
                "player": int(player) if player.isdigit() else None,
                "card": head,
                "position": position_label,   # "U" / "D" / "p<N>" / "single"
                "adj_frame":     adj_frame,
                "stable_frame":  stable_frame,
                "stable_run":    (pts[stable_start_idx][0], pts[stable_end_idx][0]),
                "stable_xy":     stable_xy,
                "final_frame":   final_frame,
                "final_xy":      final_xy,
                "final_run":     (tail[final_start_idx][0], tail[final_end_idx][0]),
                "delta_px":      delta,
                "segment":       (s_lo, s_hi),
            })

    if not events:
        return [], []

    pad = int(round(PAD_SECONDS * fps))
    win_set = set()
    for e in events:
        a = e["adj_frame"]
        if segments is not None:
            seg = _containing_segment(a, segments)
            if seg is None:
                continue
            lo, hi = seg["start_frame"], seg["end_frame"]
        else:
            lo, hi = frame_lo, frame_hi
        win_set.add((max(lo, a - pad), min(hi, a + pad)))
    if not win_set:
        return [], events
    windows = sorted(win_set)
    merged = [windows[0]]
    for s, e in windows[1:]:
        if s <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged, events


def current_segment_label(frame_idx, segments):
    for seg in segments:
        if seg["start_frame"] <= frame_idx <= seg["end_frame"]:
            lbl = seg["labels"][0] if seg.get("labels") else ""
            if lbl in LABELS_TO_RENDER:
                return lbl
    return None


def render(round_name, pred_dir, cards_dir, video_dir, output_dir):
    pred_path = os.path.join(pred_dir, f"{round_name}_predictions.json")
    cards_path = os.path.join(cards_dir, f"{round_name}_card.jsonl")
    video_path = os.path.join(video_dir, f"{round_name}.mp4")
    for p, label in [(pred_path, "prediction JSON"),
                     (cards_path, "card JSONL"),
                     (video_path, "video")]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing {label}: {p}")

    with open(pred_path) as f:
        pred = json.load(f)
    segments_to_render = [s for s in pred["timeline_segments"]
                          if s.get("labels") and s["labels"][0] in LABELS_TO_RENDER]
    if not segments_to_render:
        raise RuntimeError(f"no initial-hands 1st/2nd segments in {round_name}")

    frame_lo = min(s["start_frame"] for s in segments_to_render)
    frame_hi = max(s["end_frame"]   for s in segments_to_render)

    cards_by_frame = load_card_jsonl(cards_path)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    clips, events = detect_adjustment_clips(
        cards_by_frame, frame_lo, frame_hi, fps, segments=segments_to_render,
    )
    if not clips:
        print(f"no adjusted cards in {round_name} (delta(stable,final) <= {ADJUST_THRESH_PX}px); skipping")
        cap.release()
        return []
    print(f"adjusted cards in {round_name}:")
    for ev in events:
        sr_a, sr_b = ev["stable_run"]
        print(f"  {ev['key']}: stable={tuple(int(v) for v in ev['stable_xy'])} "
              f"(frame {ev['stable_frame']}, run {sr_a}..{sr_b}) -> "
              f"final={tuple(int(v) for v in ev['final_xy'])} (frame {ev['final_frame']}) "
              f"delta={ev['delta_px']:.1f}px, adj@frame {ev['adj_frame']}")

    # save stable-frame and final-frame snapshots for visual verification
    snap_dir = os.path.join(output_dir, f"{round_name}_frames")
    os.makedirs(snap_dir, exist_ok=True)
    for ev in events:
        for tag, fr in [("stable", ev["stable_frame"]), ("final", ev["final_frame"])]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
            ok, snap = cap.read()
            if not ok:
                continue
            cx, cy = (int(ev["stable_xy"][0]), int(ev["stable_xy"][1])) if tag == "stable" \
                else (int(ev["final_xy"][0]), int(ev["final_xy"][1]))
            color = (0, 255, 0) if tag == "stable" else (0, 0, 255)
            cv2.circle(snap, (cx, cy), 8, color, 2)
            cv2.line(snap, (0, PLAYER_MIN_Y_PX), (W, PLAYER_MIN_Y_PX), (0, 255, 255), 1)
            label = f"{ev['key']} {tag} frame {fr}  xy=({cx},{cy})"
            cv2.rectangle(snap, (0, 0), (W, 28), (0, 0, 0), -1)
            cv2.putText(snap, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1, cv2.LINE_AA)
            out = os.path.join(snap_dir, f"{ev['key']}_{tag}_f{fr}.png")
            cv2.imwrite(out, snap)
    print(f"snapshots: {snap_dir}")
    total = sum(e - s + 1 for s, e in clips)
    print(f"adjustment clips: {clips}  (total {total} frames, {total/fps:.1f}s)")

    os.makedirs(output_dir, exist_ok=True)
    tmp_path = os.path.join(output_dir, f"{round_name}_adjust_tmp.mp4")
    out_path = os.path.join(output_dir, f"{round_name}_adjust.mp4")
    writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    n_rendered = 0
    for clip_s, clip_e in clips:
        cap.set(cv2.CAP_PROP_POS_FRAMES, clip_s)
        for f_idx in range(clip_s, clip_e + 1):
            ok, frame = cap.read()
            if not ok:
                break

            seg_label = current_segment_label(f_idx, segments_to_render)
            if seg_label is None:
                continue   # skip gap frames

            # only render player-seat cards below the yellow line; no trails
            clusters = cluster_by_seat(cards_by_frame.get(f_idx, []))
            for seat in range(1, 8):
                for det in clusters.get(seat, []):
                    sd = scale_det(det)
                    if sd["center_px"][1] <= PLAYER_MIN_Y_PX:
                        continue
                    x1, y1, x2, y2 = sd["box_px"]
                    key = f"{card_key(sd)}@P{seat}"
                    color = color_for_key(key)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, key, (x1, max(0, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
                    cv2.circle(frame, sd["center_px"], 4, color, -1)

            # boundary line
            cv2.line(frame, (0, PLAYER_MIN_Y_PX), (W, PLAYER_MIN_Y_PX),
                     (0, 255, 255), 1, cv2.LINE_AA)

            # Collect every event whose adjustment frame falls inside this clip.
            events_in_clip = [ev for ev in events if clip_s <= ev["adj_frame"] <= clip_e]

            # Draw stable→final arrow + marker dots for each active event ON the frame.
            for ev in events_in_clip:
                sx, sy = int(ev["stable_xy"][0]), int(ev["stable_xy"][1])
                fx, fy = int(ev["final_xy"][0]), int(ev["final_xy"][1])
                cv2.circle(frame, (sx, sy), 8, (0, 255, 0), 2)    # green = stable
                cv2.circle(frame, (fx, fy), 8, (0, 0, 255), 2)    # red   = final
                cv2.arrowedLine(frame, (sx, sy), (fx, fy),
                                (0, 200, 255), 2, cv2.LINE_AA, tipLength=0.25)

            # Stacked banner: header + one line per event in the current clip.
            n_events = len(events_in_clip)
            banner_h = 28 + 18 * max(1, n_events)
            cv2.rectangle(frame, (0, 0), (W, banner_h), (0, 0, 0), -1)
            header = f"frame {f_idx}    {seg_label or '(gap)'}    clip {clip_s}..{clip_e}    events: {n_events}"
            cv2.putText(frame, header, (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            for i, ev in enumerate(events_in_clip):
                sx, sy = int(ev["stable_xy"][0]), int(ev["stable_xy"][1])
                fx, fy = int(ev["final_xy"][0]), int(ev["final_xy"][1])
                active = abs(f_idx - ev["adj_frame"]) <= 5
                color = (255, 255, 100) if active else (200, 200, 200)
                line = (f"  P{ev['player']} {ev['card']} ({ev['position']}): "
                        f"({sx},{sy}) -> ({fx},{fy})  d={ev['delta_px']:.1f}px  "
                        f"adj@f{ev['adj_frame']}")
                cv2.putText(frame, line, (8, 28 + 18 * (i + 1) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

            writer.write(frame)
            n_rendered += 1

    writer.release()
    cap.release()

    # re-encode to H.264 for portability
    enc = subprocess.run(
        ["ffmpeg", "-y", "-i", tmp_path, "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-preset", "medium", "-crf", "23", out_path],
        capture_output=True, text=True,
    )
    if enc.returncode == 0:
        os.remove(tmp_path)
    else:
        os.rename(tmp_path, out_path)
        print("ffmpeg failed; left raw mp4v file:", enc.stderr[-400:])

    rendered_span = f"{clips[0][0]}..{clips[-1][1]}"
    print(f"wrote {out_path}  ({n_rendered} frames, rendered {rendered_span})")
    return events


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--round", default="2025-10-08_10-29-50_028248_031683",
                    help="round name (without _predictions.json suffix)")
    ap.add_argument("--pred_dir",   default=DEFAULT_PRED_DIR)
    ap.add_argument("--cards_dir",  default=DEFAULT_CARDS_DIR)
    ap.add_argument("--video_dir",  default=DEFAULT_VIDEO_DIR)
    ap.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    args = ap.parse_args()
    render(args.round, args.pred_dir, args.cards_dir, args.video_dir, args.output_dir)


if __name__ == "__main__":
    main()
