"""Update SRSS annotation jsons' first/last keyframes from the SAM tracking.

For each split_discard_gated_srss/<stem>_annotations.json, find the matching SAM trajectory
record (by segment start/end frame) and rewrite the bbox's FIRST keyframe to the tracked
start points and the LAST keyframe to the tracked end points, in card0(red)/card1(blue)
order. Card boxes are re-centered on the tracked points (size preserved from the original
keyframe); the encompassing box / x,y,w,h and the keyframe frame are recomputed.

Segments with no matching trajectory (e.g. dropped because <2 cards were tracked) are left
unchanged. Writes annotations in place.
"""
import glob, json, os

SRSS_DIR = "/home/ubuntu/us-west-3-fs/sahithi/hard_action_annotations/full_run/split_discard_gated_srss"
TRAJ_DIR = os.path.join(SRSS_DIR, "sam_tracking", "trajectories")
IMG_W, IMG_H = 1280, 720


def _load_traj(stem):
    p = os.path.join(TRAJ_DIR, f"{stem}_srss_trajectory.jsonl")
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p) if l.strip()]


def _box_at(center, size):
    """xyxy box of given (w,h) centered at (cx,cy)."""
    cx, cy = center
    w, h = size
    return [int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)]


def _sizes_from_kf(kf):
    """(w,h) per card from the keyframe's existing card_box, fallback ~76x48."""
    out = []
    for b in (kf.get("card_box") or []):
        if b and len(b) == 4:
            out.append((b[2] - b[0], b[3] - b[1]))
        else:
            out.append((76, 48))
    while len(out) < 2:
        out.append((76, 48))
    return out


def _rewrite_kf(kf, points, sizes):
    """Set a keyframe to the two tracked `points` ([[cx,cy],[cx,cy]], card0/card1 order)."""
    boxes = [_box_at(points[i], sizes[i]) for i in range(2)]
    xs = [boxes[0][0], boxes[0][2], boxes[1][0], boxes[1][2]]
    ys = [boxes[0][1], boxes[0][3], boxes[1][1], boxes[1][3]]
    enc = [min(xs), min(ys), max(xs), max(ys)]
    kf["polygon_center"] = [[round(points[0][0], 2), round(points[0][1], 2)],
                            [round(points[1][0], 2), round(points[1][1], 2)]]
    kf["card_box"] = boxes
    kf["box"] = enc
    kf["x"] = round(enc[0] / IMG_W * 100, 3)
    kf["y"] = round(enc[1] / IMG_H * 100, 3)
    kf["width"] = round((enc[2] - enc[0]) / IMG_W * 100, 3)
    kf["height"] = round((enc[3] - enc[1]) / IMG_H * 100, 3)
    kf["color"] = ["red", "blue"]
    kf["source"] = "sam_tracking"


def main():
    ann_files = sorted(glob.glob(os.path.join(SRSS_DIR, "*_annotations.json")))
    n_updated = n_bbox = n_skip = 0
    for p in ann_files:
        d = json.load(open(p))
        stem = d["video_name"].replace(".mp4", "")
        recs = _load_traj(stem)
        if not recs:
            continue
        changed = False
        for ts in d.get("timeline_segments", []):
            # match trajectory record by segment start/end frame
            rec = next((r for r in recs
                        if r.get("start_frame") is not None
                        and abs(r["start_frame"] - ts["start_frame"]) <= 8), None)
            if rec is None or not rec.get("start_points") or not rec.get("end_points"):
                continue
            sp, ep = rec["start_points"], rec["end_points"]
            if len(sp) < 2 or len(ep) < 2 or None in sp[:2] or None in ep[:2]:
                n_skip += 1
                continue
            for bb in ts.get("bounding_boxes", []):
                kfs = bb.get("keyframes") or []
                if not kfs:
                    continue
                sizes = _sizes_from_kf(kfs[0])
                _rewrite_kf(kfs[0], sp, sizes)          # FIRST keyframe -> tracked start
                _rewrite_kf(kfs[-1], ep, sizes)         # LAST keyframe  -> tracked end
                bb["start_frame_tracked"] = rec["cards"][0]["start_point"]["frame"]
                bb["end_frame_tracked"] = rec["cards"][0]["end_point"]["frame"]
                kfs[0]["frame"] = rec["cards"][0]["start_point"]["frame"]
                kfs[-1]["frame"] = rec["cards"][0]["end_point"]["frame"]
                n_bbox += 1
                changed = True
        if changed:
            with open(p, "w") as f:
                json.dump(d, f, indent=2)
            n_updated += 1
    print(f"updated {n_updated} annotation files, {n_bbox} bboxes (first/last kf from SAM tracking, card0=red/card1=blue)")
    print(f"skipped (no valid tracked points): {n_skip}")


if __name__ == "__main__":
    main()
