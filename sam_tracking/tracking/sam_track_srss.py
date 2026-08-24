"""SAM3 dense trajectory tracking for same-rank same-suit (identical-card) splits.

For the SRSS split annotations (split_discard_gated_srss), detection-based red/blue
pairing can't tell the two identical cards apart, so we track them with SAM 3 instead:
the split segment's keyframe card_box centers become SAM point prompts (one instance per
card) and SAM propagates each bidirectionally to a dense per-frame trajectory.

The round-clip videos don't exist as files, so for each split segment we cut a temp clip
from the raw SESSION video (raw/batch_01) using the stem's frame offset (same mapping the
red/blue visualizer used), resized to 1280x720 to match the annotation coordinates.

Outputs (under split_discard_gated_srss/sam_tracking/):
  trajectories/<stem>_srss_trajectory.jsonl   one JSON line per split segment
  viz/<stem>_seg<k>.mp4                        annotated clip (trails, not red/blue boxes)
  summary.json

Usage:
  conda run -n sam3 python sam_track_srss.py [N]      # N = max videos (default all)
  env VIZ=0 to skip clips.
"""
import os, sys, glob, json, tempfile, traceback
from collections import defaultdict

import cv2
import numpy as np
import torch

from sam3.model_builder import build_sam3_video_predictor

SRSS_DIR = os.environ.get("ANN_DIR",
    "/home/ubuntu/us-west-3-fs/sahithi/hard_action_annotations/full_run/split_discard_gated_srss")
RAW_VIDEO_DIR = "/home/ubuntu/us-west-3-fs/live_dealer_blackjack/raw/batch_01"
OUT = os.environ.get("OUT_DIR", os.path.join(SRSS_DIR, "sam_tracking"))
TRAJ_SUFFIX = os.environ.get("TRAJ_SUFFIX", "_srss_trajectory")
TRAJ_DIR = os.path.join(OUT, "trajectories")
VIZ_DIR = os.path.join(OUT, "viz")
VIZ_NOTSPLIT_DIR = os.path.join(OUT, "viz_notsplit")
os.makedirs(TRAJ_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)
os.makedirs(VIZ_NOTSPLIT_DIR, exist_ok=True)

N = int(sys.argv[1]) if len(sys.argv) > 1 else None
DO_VIZ = os.environ.get("VIZ", "1") != "0"
VIZ_DROPPED = os.environ.get("VIZ_DROPPED", "0") == "1"   # review mode: viz the dropped/not-real segments only
PAD = int(os.environ.get("PAD", "6"))          # frames of context each side of the segment
W_OUT, H_OUT = 1280, 720                        # temp-clip size == annotation coord space
MOVE_THRESH = 2.5
COLORS = [(0, 0, 255), (255, 0, 0), (0, 220, 0), (0, 200, 255)]  # BGR: red=card0, blue=card1

# split verdict thresholds (same as split_trajectory_sam)
SPLIT_MIN_SEP, SPLIT_MIN_DIVERGE, SPLIT_MIN_SPREAD = 35.0, 10.0, 20.0
SPLIT_MIN_TRAVEL, SPLIT_MAX_BOX_FRAC = 15.0, 0.04
BOTH_MIN_TRAVEL = 12.0         # a card static (<this) = bystander; 12+ is real split movement
BLOWUP_FRAC = 0.02             # per-frame: drop a card's box if it exceeds this frac of frame
OUTLIER_PX = 40.0              # per-frame: drop a point straying > this from the anchor path


# ── video source (raw session + offset) ───────────────────────────────────────

def session_path_and_offset(stem):
    parts = stem.split("_")
    base = f"{parts[0]}_{parts[1]}".replace("_", " ", 1)
    clip_start = int(parts[2])
    for name in (f"{base}.mp4", f"{parts[0]}_{parts[1]}.mp4"):
        p = os.path.join(RAW_VIDEO_DIR, name)
        if os.path.exists(p):
            return p, clip_start
    return None, None


def cut_temp_clip(session_path, clip_start, lo, hi):
    """Write raw frames [(clip_start-1)+lo .. +hi] to a temp 1280x720 mp4. Returns
    (path, n_frames) or (None, 0)."""
    cap = cv2.VideoCapture(session_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, (clip_start - 1) + lo)
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W_OUT, H_OUT))
    n = 0
    for _ in range(hi - lo + 1):
        ret, fr = cap.read()
        if not ret:
            break
        if fr.shape[1] != W_OUT or fr.shape[0] != H_OUT:
            fr = cv2.resize(fr, (W_OUT, H_OUT))
        vw.write(fr)
        n += 1
    vw.release(); cap.release()
    # transcode to h264 so SAM's loader reads it reliably
    h264 = tmp + ".h264.mp4"
    rc = os.system(f'ffmpeg -y -loglevel error -i "{tmp}" -c:v libx264 -pix_fmt yuv420p '
                   f'-movflags +faststart "{h264}" 2>/dev/null')
    if rc == 0 and os.path.exists(h264):
        os.remove(tmp)
        return h264, n
    return tmp, n


# ── keyframe / prompt selection (from split_trajectory_sam) ────────────────────

def _pair_sep(cbox):
    cs = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in cbox]
    if len(cs) < 2:
        return 0.0
    return max(((cs[i][0] - cs[j][0]) ** 2 + (cs[i][1] - cs[j][1]) ** 2) ** 0.5
               for i in range(len(cs)) for j in range(i + 1, len(cs)))


def best_keyframe(seg):
    """(kf_frame_local, [card_box xyxy...], ranks, suits) from the most-separated
    keyframe with two card boxes."""
    bbs = seg.get("bounding_boxes", [])
    if not bbs:
        return None
    kfs2 = [k for k in bbs[0].get("keyframes", []) if k.get("card_box") and len(k["card_box"]) >= 2]
    if kfs2:
        kf = max(kfs2, key=lambda k: _pair_sep(k["card_box"]))
        return kf["frame"], kf["card_box"], kf.get("rank") or [], kf.get("suit") or []
    kfs = bbs[0].get("keyframes", [])
    if not kfs:
        return None
    kf = min(kfs, key=lambda k: k["frame"])
    if kf.get("card_box"):
        return kf["frame"], kf["card_box"], kf.get("rank") or [], kf.get("suit") or []
    return None


def _pt_seg_dist(px, py, ax, ay, bx, by):
    """Distance from point (px,py) to segment (ax,ay)-(bx,by)."""
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    qx, qy = ax + t * dx, ay + t * dy
    return ((px - qx) ** 2 + (py - qy) ** 2) ** 0.5


def get_keyframes(seg):
    """All keyframes (sorted by frame) that carry two card boxes — used as SAM anchors."""
    bbs = seg.get("bounding_boxes", [])
    if not bbs:
        return []
    kfs = [k for k in bbs[0].get("keyframes", [])
           if k.get("card_box") and len(k["card_box"]) >= 2]
    return sorted(kfs, key=lambda k: k["frame"])


def center_norm(b, W, H):
    x1, y1, x2, y2 = b
    return [(x1 + x2) / 2 / W, (y1 + y2) / 2 / H]


def box_center_px(bx, W, H):
    x, y, w, h = [float(v) for v in bx]
    if max(x, y, w, h) <= 1.5:
        x, w = x * W, w * W
        y, h = y * H, h * H
    return x + w / 2, y + h / 2, (x, y, w, h)


# ── verdict (from split_trajectory_sam) ────────────────────────────────────────

def _path_len(traj):
    d = 0.0
    for i in range(1, len(traj)):
        d += ((traj[i][1] - traj[i - 1][1]) ** 2 + (traj[i][2] - traj[i - 1][2]) ** 2) ** 0.5
    return d


def split_verdict(cards):
    if len(cards) < 2:
        return False, {"reason": "fewer than 2 cards tracked"}
    a = {t[0]: (t[1], t[2]) for t in cards[0]["trajectory"]}
    b = {t[0]: (t[1], t[2]) for t in cards[1]["trajectory"]}
    common = sorted(set(a) & set(b))
    if not common:
        return False, {"reason": "no overlapping frames"}
    seps = [(((a[f][0] - b[f][0]) ** 2 + (a[f][1] - b[f][1]) ** 2) ** 0.5) for f in common]
    k = max(1, min(3, len(seps)))
    sep_start, sep_end = sum(seps[:k]) / k, sum(seps[-k:]) / k
    sep_max, sep_min = max(seps), min(seps)
    spread, diverge = sep_max - sep_min, sep_end - sep_start
    travels = [_path_len(c["trajectory"]) for c in cards]
    max_travel, min_travel = max(travels), min(travels)
    max_box_frac = max((c.get("max_box_frac") or 0.0) for c in cards)
    stats = {"sep_start": round(sep_start, 1), "sep_end": round(sep_end, 1),
             "sep_max": round(sep_max, 1), "sep_min": round(sep_min, 1),
             "spread": round(spread, 1), "diverge": round(diverge, 1),
             "max_travel": round(max_travel, 1), "min_travel": round(min_travel, 1),
             "max_box_frac": round(max_box_frac, 4)}
    if max_box_frac > SPLIT_MAX_BOX_FRAC:
        return False, {**stats, "reason": "track blew up"}
    if max_travel < SPLIT_MIN_TRAVEL and spread < SPLIT_MIN_SPREAD:
        return False, {**stats, "reason": "cards static"}
    # Require BOTH cards to move — a static card means a wrong/mis-paired card (a static
    # neighbour, or ten-value drawn cards picked instead of the real split pair). Only keep
    # clearly-valid splits where both cards are actually moved apart.
    if min_travel < BOTH_MIN_TRAVEL:
        return False, {**stats, "reason": "one card static (wrong/mis-paired)"}
    if sep_max < SPLIT_MIN_SEP:
        return False, {**stats, "reason": "cards never separate"}
    # For SRSS (same rank AND suit), two distinct cards tracked at one seat MUST be a split
    # — you can't hold two identical cards in one hand. So once we have: 2 cards, not blown
    # up, both moving (rules out a static bystander/cross-player card), and they REACH a real
    # gap (sep_max >= SPLIT_MIN_SEP, rules out one card tracked as two) — it's a split. The
    # end/stay-apart distance is NOT required: split hands often drift back closer as more
    # cards are dealt onto them (e.g. sep peaks at 88 then settles to ~38).
    return True, {**stats, "reason": "ok"}


# ── track one segment on its temp clip ─────────────────────────────────────────

def order_cards(cards):
    """Label the two tracked cards red/blue WITHOUT reordering. SAM object ids come from
    the annotation's card_box order (obj_id i is prompted from card_box[i]), and the
    annotation already stores index 0 = upper card at start, carried consistently to the
    end keyframe. Keeping that order means the trajectory's card0/card1 stay aligned with
    the annotation's card0/card1 — no swapping before vs after tracking. index 0 = 'red'
    (upper), index 1 = 'blue' (lower)."""
    cards = sorted(cards, key=lambda c: c["card_index"])   # by SAM obj_id (= annotation order)
    for i, c in enumerate(cards):
        c["_sam_oid"] = c["card_index"]   # original SAM id, for viz box lookup
        c["card_index"] = i
        c["color"] = "red" if i == 0 else ("blue" if i == 1 else "other")
    return cards


def _n_tracked(per_card):
    return sum(1 for pts in per_card.values() if pts)


def _run_track(predictor, sid, anchors, prop_start, start_t, end_t, num_frames, W, H):
    """Prompt each card with one positive point at each anchor frame, propagate both ways,
    collect per-frame boxes (blown-up frames dropped).
    anchors = [(frame_t, [center_per_card, ...]), ...]."""
    predictor.handle_request(dict(type="reset_session", session_id=sid))
    for at, acenters in anchors:
        for oid, c in enumerate(acenters):
            predictor.handle_request(dict(type="add_prompt", session_id=sid, frame_index=at,
                                          obj_id=oid, points=[c], point_labels=[1],
                                          rel_coordinates=True))
    max_track = (end_t - start_t) + 3
    per_card = defaultdict(list)
    boxes_by_frame = defaultdict(dict)
    max_area = defaultdict(float)
    for r in predictor.handle_stream_request(dict(
            type="propagate_in_video", session_id=sid, start_frame_index=prop_start,
            max_frame_num_to_track=max_track, propagation_direction="both")):
        fi = r["frame_index"]
        if not (start_t <= fi <= end_t):
            continue
        o = r["outputs"]
        ids = o["out_obj_ids"]; ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        obx = o["out_boxes_xywh"]
        probs = o.get("out_probs")
        probs = probs.tolist() if hasattr(probs, "tolist") else (list(probs) if probs is not None else None)
        for j, oid in enumerate(ids):
            bx = obx[j]; bx = bx.tolist() if hasattr(bx, "tolist") else list(bx)
            cx, cy, xywh = box_center_px(bx, W, H)
            # Drop ONLY blown-up frames (box >5x a card); good tracks never hit this.
            if (xywh[2] * xywh[3]) / (W * H) > BLOWUP_FRAC:
                continue
            p = float(probs[j]) if probs is not None and j < len(probs) else None
            per_card[int(oid)].append([fi, round(cx, 1), round(cy, 1), round(p, 3) if p is not None else None])
            boxes_by_frame[fi][int(oid)] = xywh
            max_area[int(oid)] = max(max_area[int(oid)], xywh[2] * xywh[3])
    return per_card, boxes_by_frame, max_area


def process_segment(predictor, sid, seg, lo, W, H, num_frames):
    """seg frames are LOCAL (clip-relative); the temp clip starts at `lo`, so temp index
    = local - lo. Returns (rec_local, boxes_by_temp_frame) or None."""
    kf = best_keyframe(seg)
    if kf is None:
        return None
    kf_local, cboxes, ranks, suits = kf
    kf_t = int(min(max(kf_local - lo, 0), num_frames - 1))
    start_t = int(max(seg["start_frame"] - lo, 0))
    end_t = int(min(seg["end_frame"] - lo, num_frames - 1))
    centers = [center_norm(b, W, H) for b in cboxes]

    # PRIMARY: one positive point per card at the most-separated keyframe (best general
    # tracking; keeps the two identical cards' masks distinct without perturbation).
    per_card, boxes_by_frame, max_area = _run_track(
        predictor, sid, [(kf_t, centers)], kf_t, start_t, end_t, num_frames, W, H)
    # FALLBACK: if SAM lost a card (<2 tracked), retry with double-anchor prompting (pin
    # BOTH keyframes). Only triggers on these hard identical-card cases, so it never
    # touches the good tracks that already succeed with the single prompt.
    if _n_tracked(per_card) < 2:
        akfs = get_keyframes(seg)
        if len(akfs) >= 2:
            anchors = [(int(min(max(a["frame"] - lo, 0), num_frames - 1)),
                        [center_norm(b, W, H) for b in a["card_box"]]) for a in akfs]
            p2, b2, m2 = _run_track(predictor, sid, anchors, min(a[0] for a in anchors),
                                    start_t, end_t, num_frames, W, H)
            if _n_tracked(p2) >= _n_tracked(per_card):
                per_card, boxes_by_frame, max_area = p2, b2, m2

    cards = []
    for oid in sorted(per_card):
        pts = sorted(per_card[oid], key=lambda t: t[0])
        traj, prev = [], None
        for (f, cx, cy, p) in pts:
            mv = 0
            if prev is not None:
                mv = 1 if ((cx - prev[0]) ** 2 + (cy - prev[1]) ** 2) ** 0.5 > MOVE_THRESH else 0
            traj.append([f, cx, cy, mv, p])   # temp-frame index
            prev = (cx, cy)
        rank = ranks[oid] if oid < len(ranks) else None
        suit = suits[oid] if oid < len(suits) else None
        cards.append({
            "card_index": oid, "rank": rank, "suit": suit,
            "final_class": (str(rank) + str(suit)) if rank and suit else None,
            "prompt_point_norm": centers[oid] if oid < len(centers) else None,
            "n_frames": len(traj), "max_box_frac": round(max_area[oid] / (W * H), 4),
            "trajectory": traj,
        })
    is_split, sep_stats = split_verdict(cards)
    cards = order_cards(cards)   # index 0 = upper card at start (red), 1 = lower (blue)
    # build record with LOCAL frames for saving
    cards_local = []
    for c in cards:
        cl = dict(c)
        cl.pop("_sam_oid", None)   # internal-only; keep the saved json clean
        tj = [[t[0] + lo, t[1], t[2], t[3], t[4]] for t in c["trajectory"]]
        cl["trajectory"] = tj
        # explicit start/end points of this card's tracked trajectory (card0/card1 order)
        if tj:
            cl["start_point"] = {"frame": tj[0][0], "cx": tj[0][1], "cy": tj[0][2]}
            cl["end_point"] = {"frame": tj[-1][0], "cx": tj[-1][1], "cy": tj[-1][2]}
        else:
            cl["start_point"] = cl["end_point"] = None
        cards_local.append(cl)
    # ordered [card0, card1] start/end points (like the before/after red-blue keyframes)
    start_points = [[c["start_point"]["cx"], c["start_point"]["cy"]] if c.get("start_point") else None
                    for c in cards_local]
    end_points = [[c["end_point"]["cx"], c["end_point"]["cy"]] if c.get("end_point") else None
                  for c in cards_local]
    rec_local = {
        "segment_id": seg.get("id"), "action": "split", "is_split": is_split,
        "split_stats": sep_stats, "start_frame": start_t + lo, "end_frame": end_t + lo,
        "prompt_frame": kf_t + lo, "resolution": f"{W}x{H}", "n_cards": len(cards),
        "start_points": start_points, "end_points": end_points,
        "cards": cards_local,
    }
    # temp-space rec for viz
    rec_temp = {"start_frame": start_t, "end_frame": end_t, "is_split": is_split,
                "split_stats": sep_stats, "cards": cards}
    return rec_local, rec_temp, boxes_by_frame


def render_viz(clip_path, rec_temp, boxes_by_frame, out_path):
    start_f, end_f = rec_temp["start_frame"], rec_temp["end_frame"]
    cap = cv2.VideoCapture(clip_path)
    W = int(cap.get(3)); H = int(cap.get(4)); fps = cap.get(5) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    tmp = out_path + ".tmp.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    trails = defaultdict(list)
    for f in range(start_f, end_f + 1):
        ret, img = cap.read()
        if not ret:
            break
        for card in rec_temp["cards"]:
            for (ff, cx, cy, mv, p) in card["trajectory"]:
                if ff == f:
                    trails[card["card_index"]].append((int(cx), int(cy)))
        for card in rec_temp["cards"]:
            idx = card["card_index"]; color = COLORS[idx % len(COLORS)]
            box_oid = card.get("_sam_oid", idx)     # box stream keyed by original SAM id
            xywh = boxes_by_frame.get(f, {}).get(box_oid)
            if xywh:
                x, y, w, h = xywh
                cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
                lbl = f"card{idx}" + (f" {card['final_class']}" if card['final_class'] else "")
                cv2.putText(img, lbl, (int(x), max(0, int(y) - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            pts = trails[idx]
            for k in range(1, len(pts)):
                cv2.line(img, pts[k - 1], pts[k], color, 2)
            if pts:
                cv2.circle(img, pts[-1], 3, color, -1)
        v = "SPLIT" if rec_temp["is_split"] else "NOT SPLIT"
        vc = (0, 220, 0) if rec_temp["is_split"] else (0, 0, 255)
        ss = rec_temp["split_stats"]
        cv2.putText(img, f"SAM3 srss  {v}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, vc, 2)
        cv2.putText(img, f"sep {ss.get('sep_start')}->{ss.get('sep_end')} max {ss.get('sep_max')}px",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, vc, 2)
        vw.write(img)
    vw.release(); cap.release()
    rc = os.system(f'ffmpeg -y -loglevel error -i "{tmp}" -c:v libx264 -pix_fmt yuv420p '
                   f'-movflags +faststart "{out_path}" 2>/dev/null')
    if rc == 0 and os.path.exists(out_path):
        os.remove(tmp)
    else:
        os.replace(tmp, out_path)


def main():
    ann_files = sorted(glob.glob(os.path.join(SRSS_DIR, "*_annotations.json")))
    if N:
        ann_files = ann_files[:N]
    print(f"[info] {len(ann_files)} SRSS annotation files, VIZ={DO_VIZ}", flush=True)
    print("[info] building SAM3 video predictor ...", flush=True)
    predictor = build_sam3_video_predictor()

    stats = {"videos": 0, "segments": 0, "split": 0, "not_split": 0,
             "dropped_1card": 0, "dropped_viz": 0, "failed": []}
    for vi, jf in enumerate(ann_files):
        d = json.load(open(jf))
        stem = d["video_name"].replace(".mp4", "")
        segs = [s for s in d.get("timeline_segments", []) if s.get("bounding_boxes")]
        if not segs:
            continue
        traj_path = os.path.join(TRAJ_DIR, f"{stem}{TRAJ_SUFFIX}.jsonl")
        # resume: skip when trajectory exists (real split already done). In VIZ_DROPPED
        # review mode we ONLY want the dropped videos, so skip any that have a trajectory.
        if os.path.exists(traj_path) and (VIZ_DROPPED or not DO_VIZ):
            print(f"[skip] {stem} (exists)", flush=True)
            continue
        sess, clip_start = session_path_and_offset(stem)
        if sess is None:
            stats["failed"].append({"video": stem, "error": "session video not found"})
            continue
        try:
            lines = []
            for k, seg in enumerate(segs):
                lo = max(0, int(seg["start_frame"]) - PAD)
                hi = int(seg["end_frame"]) + PAD
                clip, nf = cut_temp_clip(sess, clip_start, lo, hi)
                if clip is None or nf == 0:
                    continue
                try:
                    sid = predictor.handle_request(dict(type="start_session", resource_path=clip))["session_id"]
                    res = process_segment(predictor, sid, seg, lo, W_OUT, H_OUT, nf)
                    predictor.handle_request(dict(type="close_session", session_id=sid))
                    if res is None:
                        continue
                    rec_local, rec_temp, boxes = res
                    rec_local["video"] = stem
                    stats["segments"] += 1
                    n_tracked = sum(1 for c in rec_local["cards"] if c.get("trajectory"))
                    is_real = (n_tracked >= 2) and rec_local["is_split"]
                    if VIZ_DROPPED:
                        # review mode: render ONLY the dropped (not-real) segments to a
                        # separate viz dir; never write trajectories here.
                        if not is_real:
                            reason = rec_local["split_stats"].get("reason", "?") if n_tracked >= 2 else "fewer than 2 cards tracked"
                            render_viz(clip, rec_temp, boxes,
                                       os.path.join(VIZ_NOTSPLIT_DIR, f"{stem}_seg{k}.mp4"))
                            stats["dropped_viz"] += 1
                        continue
                    # normal mode: keep only REAL splits (both cards move + separate).
                    if n_tracked < 2:
                        stats["dropped_1card"] += 1
                        continue
                    if not rec_local["is_split"]:
                        stats["not_split"] += 1
                        continue
                    stats["split"] += 1
                    lines.append(rec_local)
                    if DO_VIZ:
                        render_viz(clip, rec_temp, boxes,
                                   os.path.join(VIZ_DIR, f"{stem}_seg{k}.mp4"))
                finally:
                    if clip and os.path.exists(clip):
                        os.remove(clip)
            if not lines:   # nothing tracked with 2 cards -> don't write an empty file
                print(f"[{vi+1}/{len(ann_files)}] {stem}: 0 seg(s) kept (all <2 cards)", flush=True)
                continue
            with open(traj_path, "w") as fh:
                for rec in lines:
                    fh.write(json.dumps(rec) + "\n")
            stats["videos"] += 1
            print(f"[{vi+1}/{len(ann_files)}] {stem}: {len(lines)} seg(s) -> {traj_path}", flush=True)
        except Exception as e:
            stats["failed"].append({"video": stem, "error": str(e)})
            print(f"[FAIL] {stem}: {e}", flush=True)
            traceback.print_exc()

    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    print("\n===== DONE =====")
    print(json.dumps({k: v for k, v in stats.items() if k != "failed"}, indent=2))
    print(f"failed: {len(stats['failed'])}")


if __name__ == "__main__":
    main()
