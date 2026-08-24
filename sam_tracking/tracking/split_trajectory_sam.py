"""Split-case card-trajectory extraction with SAM 3.

For every 'split' action segment in
  /home/ubuntu/us-west-3-fs/live_dealer_blackjack/action_annotation/split_updated
we take the detector's per-card keyframe boxes (card_box) and use their centers
as SAM 3 tracker point-prompts (one instance per card). SAM propagates each card
bidirectionally across the segment to produce a dense per-frame trajectory of the
two cards as they separate.

Outputs (in /home/ubuntu/sahithi/split_sam_output):
  trajectories/<stem>_split_trajectory.jsonl   one JSON line per split segment
  viz/<stem>_seg<k>.mp4                         annotated segment clip
  summary.json                                  run-level stats

Usage:
  python split_trajectory_sam.py [N] [START]
    N     : max number of videos to process (default: all found)
    START : index into the sorted found-video list to start at (default 0)
  env VIZ=0 to skip rendering annotated clips.
"""
import os, sys, glob, json, traceback
from collections import defaultdict

import cv2
import numpy as np
import torch

from sam3.model_builder import build_sam3_video_predictor

ANN_DIR = os.environ.get(
    "SPLIT_ANN_DIR",
    "/home/ubuntu/us-west-3-fs/live_dealer_blackjack/action_annotation/split_updated")
# If set, keep only segments whose keyframe has two identical cards (same rank AND suit).
SAME_RANK_SUIT = os.environ.get("SAME_RANK_SUIT", "0") == "1"
# If set, only write segments the trajectory verdict accepts as real splits; rejected
# ones are logged (with reason) to dropped.jsonl and their viz goes to viz_dropped/.
DROP_NONSPLIT = os.environ.get("DROP_NONSPLIT", "0") == "1"
ROOTS = [
    "/home/ubuntu/us-west-3-fs/live_dealer_blackjack/roundcut/good_quality_rounds",
    "/home/ubuntu/us-west-3-fs/live_dealer_blackjack/roundcut/multidealer_300rounds",
    "/home/ubuntu/us-west-3-fs/live_dealer_blackjack/roundcut/remaining_vid",
    "/home/ubuntu/us-west-3-fs/live_dealer_blackjack/roundcut/good_quality_rounds_sparse_action_retrieved",
]
OUT = os.environ.get("SPLIT_OUT", "/home/ubuntu/sahithi/split_sam_output")
TRAJ_DIR = os.path.join(OUT, "trajectories")
VIZ_DIR = os.path.join(OUT, "viz")
os.makedirs(TRAJ_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)

N = int(sys.argv[1]) if len(sys.argv) > 1 else None
START = int(sys.argv[2]) if len(sys.argv) > 2 else 0
DO_VIZ = os.environ.get("VIZ", "1") != "0"
MOVE_THRESH = 2.5  # px/frame -> moving flag
COLORS = [(0, 0, 255), (0, 220, 0), (255, 90, 0), (0, 200, 255)]


def build_video_index():
    idx = {}
    for r in ROOTS:
        for p in glob.glob(os.path.join(r, "*.mp4")):
            idx.setdefault(os.path.basename(p), p)
    return idx


def seg_is_same_rank_suit(seg):
    """True if any keyframe carries two identical cards (same rank AND suit)."""
    for bb in seg.get("bounding_boxes", []):
        for kf in bb.get("keyframes", []):
            r = kf.get("rank") or []
            s = kf.get("suit") or []
            if len(r) >= 2 and len(s) >= 2 and r[0] and s[0] and r[0] == r[1] and s[0] == s[1]:
                return True
    return False


def load_jobs(idx):
    jobs = []
    for jf in sorted(glob.glob(os.path.join(ANN_DIR, "*.json"))):
        d = json.load(open(jf))
        segs = d.get("timeline_segments", [])
        if SAME_RANK_SUIT:
            segs = [s for s in segs if seg_is_same_rank_suit(s)]
        if not segs:
            continue
        vname = d["video_name"]
        if vname not in idx:
            continue
        jobs.append((vname, idx[vname], segs))
    return jobs


def _pair_sep(cbox):
    """Max pairwise distance between card_box centers (0 if <2 boxes)."""
    cs = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in cbox]
    if len(cs) < 2:
        return 0.0
    return max(((cs[i][0] - cs[j][0]) ** 2 + (cs[i][1] - cs[j][1]) ** 2) ** 0.5
               for i in range(len(cs)) for j in range(i + 1, len(cs)))


def _label_kf(kfs):
    """Pick the keyframe to read rank/suit labels from. The detector often misreads a
    card at the separated frame (occluded/rotated mid-split) — and sometimes reads the
    wrong label with HIGH confidence — so the prompt keyframe is a poor label source.
    Prefer keyframes that form a valid pair (both cards same rank; and same suit too in
    SAME_RANK_SUIT mode), then take the most confident of those."""
    def score(k):
        c = k.get("conf") or []
        return min(c) if c else 0.0

    def valid(k):
        r = k.get("rank") or []
        s = k.get("suit") or []
        if len(r) < 2 or not r[0] or r[0] != r[1]:
            return False
        if SAME_RANK_SUIT and (len(s) < 2 or not s[0] or s[0] != s[1]):
            return False
        return True

    pref = [k for k in kfs if valid(k)]
    return max(pref or kfs, key=score)


def best_keyframe(seg):
    """Return (kf_frame, [card_box_xyxy...], ranks, suits).
    kf_frame + boxes come from the MOST-SEPARATED keyframe (so the two point prompts
    latch onto different cards, not one stacked pair). rank/suit come from a valid,
    most-confident keyframe (decoupled — the separated frame often mislabels a card)."""
    bbs = seg.get("bounding_boxes", [])
    if not bbs:
        return None
    kfs2 = [k for k in bbs[0].get("keyframes", []) if k.get("card_box") and len(k["card_box"]) >= 2]
    if kfs2:
        kf = max(kfs2, key=lambda k: _pair_sep(k["card_box"]))
        lab = _label_kf(kfs2)
        return kf["frame"], kf["card_box"], lab.get("rank") or [], lab.get("suit") or []
    # no keyframe with 2 boxes: use earliest keyframe's card_box or pair box
    kfs = bbs[0].get("keyframes", [])
    if not kfs:
        return None
    kf = min(kfs, key=lambda k: k["frame"])
    if kf.get("card_box"):
        return kf["frame"], kf["card_box"], kf.get("rank") or [], kf.get("suit") or []
    if kf.get("box"):
        return kf["frame"], [kf["box"]], kf.get("rank") or [], kf.get("suit") or []
    return None


def center_norm(b, W, H):
    x1, y1, x2, y2 = b
    return [(x1 + x2) / 2 / W, (y1 + y2) / 2 / H]


def box_center_px(bx, W, H):
    x, y, w, h = [float(v) for v in bx]
    if max(x, y, w, h) <= 1.5:  # relative coords
        x, w = x * W, w * W
        y, h = y * H, h * H
    return x + w / 2, y + h / 2, (x, y, w, h)


SPLIT_MIN_SEP = 40.0      # px: two cards must reach at least ~a card-width apart
SPLIT_MIN_DIVERGE = 10.0  # px: separation must grow from start to end
SPLIT_MIN_SPREAD = 20.0   # px: gap must open up (sep_max - sep_min) -> real motion, not static
SPLIT_MIN_TRAVEL = 15.0   # px: at least one card must physically travel this far
SPLIT_MAX_BOX_FRAC = 0.04 # if a card's box ever exceeds this frac of frame -> track blew up


def _path_len(traj):
    d = 0.0
    for i in range(1, len(traj)):
        d += ((traj[i][1] - traj[i - 1][1]) ** 2 + (traj[i][2] - traj[i - 1][2]) ** 2) ** 0.5
    return d


def split_verdict(cards):
    """Decide whether the tracked trajectories actually show a genuine split:
    two distinct cards that (a) physically move, (b) spread apart, and (c) don't
    blow up onto a large non-card region. Returns (is_split, stats)."""
    base = {"sep_start": None, "sep_end": None, "sep_max": None, "sep_min": None,
            "spread": None, "diverge": None, "max_travel": None, "max_box_frac": None}
    if len(cards) < 2:
        return False, {**base, "reason": "fewer than 2 cards tracked"}
    a = {t[0]: (t[1], t[2]) for t in cards[0]["trajectory"]}
    b = {t[0]: (t[1], t[2]) for t in cards[1]["trajectory"]}
    common = sorted(set(a) & set(b))
    if not common:
        return False, {**base, "reason": "no overlapping frames"}
    seps = [(((a[f][0] - b[f][0]) ** 2 + (a[f][1] - b[f][1]) ** 2) ** 0.5) for f in common]
    k = max(1, min(3, len(seps)))
    sep_start = sum(seps[:k]) / k
    sep_end = sum(seps[-k:]) / k
    sep_max, sep_min = max(seps), min(seps)
    spread = sep_max - sep_min
    diverge = sep_end - sep_start
    max_travel = max(_path_len(c["trajectory"]) for c in cards)
    max_box_frac = max((c.get("max_box_frac") or 0.0) for c in cards)

    stats = {"sep_start": round(sep_start, 1), "sep_end": round(sep_end, 1),
             "sep_max": round(sep_max, 1), "sep_min": round(sep_min, 1),
             "spread": round(spread, 1), "diverge": round(diverge, 1),
             "max_travel": round(max_travel, 1), "max_box_frac": round(max_box_frac, 4)}

    if max_box_frac > SPLIT_MAX_BOX_FRAC:
        return False, {**stats, "reason": "track blew up (box too large / not a card)"}
    if max_travel < SPLIT_MIN_TRAVEL and spread < SPLIT_MIN_SPREAD:
        return False, {**stats, "reason": "cards static (no split motion in this window)"}
    if sep_max < SPLIT_MIN_SEP:
        return False, {**stats, "reason": "cards never separate (tracks overlap)"}
    if spread < SPLIT_MIN_SPREAD or diverge < SPLIT_MIN_DIVERGE:
        return False, {**stats, "reason": "cards do not move apart over segment"}
    return True, {**stats, "reason": "ok"}


def process_segment(predictor, sid, seg, W, H, num_frames):
    kf = best_keyframe(seg)
    if kf is None:
        return None
    kf_frame, cboxes, ranks, suits = kf
    kf_frame = int(min(max(kf_frame, 0), num_frames - 1))
    start_f = int(max(seg["start_frame"], 0))
    end_f = int(min(seg["end_frame"], num_frames - 1))

    predictor.handle_request(dict(type="reset_session", session_id=sid))

    centers = [center_norm(b, W, H) for b in cboxes]
    for oid, c in enumerate(centers):
        predictor.handle_request(dict(
            type="add_prompt", session_id=sid, frame_index=kf_frame,
            obj_id=oid, points=[c], point_labels=[1], rel_coordinates=True))

    span_back = kf_frame - start_f
    span_fwd = end_f - kf_frame
    max_track = max(span_back, span_fwd) + 3

    per_card = defaultdict(list)      # oid -> [[frame, cx, cy, prob], ...]
    boxes_by_frame = defaultdict(dict)  # frame -> {oid: (x,y,w,h)}
    max_area = defaultdict(float)     # oid -> max box area (px^2)
    for r in predictor.handle_stream_request(dict(
            type="propagate_in_video", session_id=sid,
            start_frame_index=kf_frame, max_frame_num_to_track=max_track,
            propagation_direction="both")):
        fi = r["frame_index"]
        if not (start_f <= fi <= end_f):
            continue
        o = r["outputs"]
        ids = o["out_obj_ids"]
        ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        obx = o["out_boxes_xywh"]
        probs = o.get("out_probs")
        probs = probs.tolist() if hasattr(probs, "tolist") else (list(probs) if probs is not None else None)
        for j, oid in enumerate(ids):
            bx = obx[j]
            bx = bx.tolist() if hasattr(bx, "tolist") else list(bx)
            cx, cy, xywh = box_center_px(bx, W, H)
            p = float(probs[j]) if probs is not None and j < len(probs) else None
            per_card[int(oid)].append([fi, round(cx, 1), round(cy, 1),
                                       round(p, 3) if p is not None else None])
            boxes_by_frame[fi][int(oid)] = xywh
            max_area[int(oid)] = max(max_area[int(oid)], xywh[2] * xywh[3])

    cards = []
    for oid in sorted(per_card):
        pts = sorted(per_card[oid], key=lambda t: t[0])
        # moving flag from frame-to-frame displacement
        traj = []
        prev = None
        for (f, cx, cy, p) in pts:
            mv = 0
            if prev is not None:
                d = ((cx - prev[0]) ** 2 + (cy - prev[1]) ** 2) ** 0.5
                mv = 1 if d > MOVE_THRESH else 0
            traj.append([f, cx, cy, mv, p])
            prev = (cx, cy)
        rank = ranks[oid] if oid < len(ranks) else None
        suit = suits[oid] if oid < len(suits) else None
        fc = (str(rank) + str(suit)) if rank and suit else None
        cards.append({
            "card_index": oid,
            "rank": rank, "suit": suit, "final_class": fc,
            "prompt_point_norm": centers[oid] if oid < len(centers) else None,
            "prompt_box_xyxy": cboxes[oid] if oid < len(cboxes) else None,
            "n_frames": len(traj),
            "max_box_frac": round(max_area[oid] / (W * H), 4),
            "trajectory": traj,  # [frame, cx, cy, moving_flag, prob]
        })

    is_split, sep_stats = split_verdict(cards)
    rec = {
        "video": None,  # filled by caller
        "segment_id": seg.get("id"),
        "action": "split",
        "is_split": is_split,          # does the SAM trajectory actually show a split?
        "split_stats": sep_stats,      # sep_start/sep_end/sep_max/diverge (px) + reason
        "start_frame": start_f,
        "end_frame": end_f,
        "prompt_frame": kf_frame,
        "resolution": f"{W}x{H}",
        "n_cards": len(cards),
        "cards": cards,
    }
    return rec, boxes_by_frame


def render_viz(video_path, rec, boxes_by_frame, out_path):
    start_f, end_f = rec["start_frame"], rec["end_frame"]
    cap = cv2.VideoCapture(video_path)
    W = int(cap.get(3)); H = int(cap.get(4)); fps = cap.get(5) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    tmp_path = out_path + ".tmp.mp4"
    vw = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    trails = defaultdict(list)
    for f in range(start_f, end_f + 1):
        ret, frame = cap.read()
        if not ret:
            break
        img = frame
        # accumulate trails
        for card in rec["cards"]:
            for (ff, cx, cy, mv, p) in card["trajectory"]:
                if ff == f:
                    trails[card["card_index"]].append((int(cx), int(cy)))
        for card in rec["cards"]:
            oid = card["card_index"]
            color = COLORS[oid % len(COLORS)]
            xywh = boxes_by_frame.get(f, {}).get(oid)
            if xywh:
                x, y, w, h = xywh
                cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
                lbl = f"card{oid}" + (f" {card['final_class']}" if card['final_class'] else "")
                cv2.putText(img, lbl, (int(x), max(0, int(y) - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            pts = trails[oid]
            for k in range(1, len(pts)):
                cv2.line(img, pts[k - 1], pts[k], color, 2)
            if pts:
                cv2.circle(img, pts[-1], 3, color, -1)
        verdict = "SPLIT" if rec["is_split"] else "NOT SPLIT"
        vcol = (0, 220, 0) if rec["is_split"] else (0, 0, 255)
        ss = rec["split_stats"]
        cv2.putText(img, f"f{f} [{start_f}-{end_f}]  {verdict}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, vcol, 2)
        cv2.putText(img, f"sep {ss.get('sep_start')}->{ss.get('sep_end')} max {ss.get('sep_max')}px",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, vcol, 2)
        vw.write(img)
    vw.release()
    cap.release()
    # transcode mp4v -> H.264 (broadly viewable); fall back to raw mp4v on failure
    rc = os.system(f'ffmpeg -y -loglevel error -i "{tmp_path}" -c:v libx264 '
                   f'-pix_fmt yuv420p -movflags +faststart "{out_path}" 2>/dev/null')
    if rc == 0 and os.path.exists(out_path):
        os.remove(tmp_path)
    else:
        os.replace(tmp_path, out_path)


def main():
    idx = build_video_index()
    jobs = load_jobs(idx)
    print(f"[info] {len(jobs)} split videos available (of matched)", flush=True)
    jobs = jobs[START:(START + N) if N else None]
    print(f"[info] processing {len(jobs)} videos (START={START}, N={N}), VIZ={DO_VIZ}", flush=True)

    print("[info] building SAM3 video predictor ...", flush=True)
    predictor = build_sam3_video_predictor()

    viz_drop_dir = os.path.join(OUT, "viz_dropped")
    dropped_path = os.path.join(OUT, "dropped.jsonl")
    if DROP_NONSPLIT:
        os.makedirs(viz_drop_dir, exist_ok=True)

    stats = {"videos": 0, "segments": 0, "kept": 0, "dropped": 0, "cards": 0, "failed": []}
    for vi, (vname, vpath, segs) in enumerate(jobs):
        stem = os.path.splitext(vname)[0]
        traj_path = os.path.join(TRAJ_DIR, f"{stem}_split_trajectory.jsonl")
        if os.path.exists(traj_path):
            print(f"[skip] {stem} (exists)", flush=True)
            continue
        try:
            cap = cv2.VideoCapture(vpath)
            W = int(cap.get(3)); H = int(cap.get(4)); num_frames = int(cap.get(7))
            cap.release()
            sid = predictor.handle_request(dict(type="start_session", resource_path=vpath))["session_id"]
            lines, drops = [], []
            for k, seg in enumerate(segs):
                res = process_segment(predictor, sid, seg, W, H, num_frames)
                if res is None:
                    print(f"   [warn] {stem} seg{k}: no usable keyframe, skipped", flush=True)
                    continue
                rec, boxes_by_frame = res
                rec["video"] = stem
                stats["segments"] += 1
                keep = rec["is_split"] or not DROP_NONSPLIT
                if rec["is_split"]:
                    stats["kept"] += 1
                else:
                    stats["dropped"] += 1
                if keep:
                    lines.append(rec)
                    stats["cards"] += rec["n_cards"]
                else:
                    drops.append(rec)
                if DO_VIZ:
                    vdir = VIZ_DIR if keep else viz_drop_dir
                    render_viz(vpath, rec, boxes_by_frame,
                               os.path.join(vdir, f"{stem}_seg{k}.mp4"))
            predictor.handle_request(dict(type="close_session", session_id=sid))
            with open(traj_path, "w") as fh:
                for rec in lines:
                    fh.write(json.dumps(rec) + "\n")
            if drops:
                with open(dropped_path, "a") as fh:
                    for rec in drops:
                        fh.write(json.dumps({"video": rec["video"], "segment_id": rec["segment_id"],
                                             "reason": rec["split_stats"].get("reason"),
                                             "split_stats": rec["split_stats"]}) + "\n")
            stats["videos"] += 1
            tag = f"{len(lines)} kept" + (f", {len(drops)} dropped" if drops else "")
            print(f"[{vi+1}/{len(jobs)}] {stem}: {tag} -> {traj_path}", flush=True)
        except Exception as e:
            stats["failed"].append({"video": stem, "error": str(e)})
            print(f"[FAIL] {stem}: {e}", flush=True)
            traceback.print_exc()
            try:
                predictor.handle_request(dict(type="close_session", session_id=sid))
            except Exception:
                pass

    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    print("\n===== DONE =====")
    print(json.dumps({k: v for k, v in stats.items() if k != "failed"}, indent=2))
    print(f"failed: {len(stats['failed'])}")


if __name__ == "__main__":
    main()
