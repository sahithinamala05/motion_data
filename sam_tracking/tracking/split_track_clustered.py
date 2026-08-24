"""Split-card SAM tracking, seeded from the EXISTING card detections.

Instead of seeding SAM on the annotation's card_box (which drifts / mislabels on the
hard identical-card splits), we:
  1. Read the pre-computed card detections (rank/suit/center) for the round-cut video
     from card_detections_roundwise (1920x1080 space, local 0-indexed frames).
  2. In the split segment, find the SAME-RANK pair near the annotated split location
     (two cards of the same normalized rank, close together, in the player zone).
  3. Seed SAM on those two card centers and track them bidirectionally.

Rank label comes from the detection (reliable); suits come from the pair's detections.

Output (SPLIT_OUT, default .../split_sam_output/clustered):
  trajectories/<stem>_split_trajectory.jsonl , viz/<stem>_seg<k>.mp4 , summary.json

Env: SPLIT_ANN_DIR, SPLIT_OUT, SAME_RANK_SUIT=1 (identical-card segments only),
     VIZ=0, N / START via argv.
"""
import os, sys, glob, json, math, traceback
from collections import defaultdict

import cv2
import numpy as np
import torch

from sam3.model_builder import build_sam3_video_predictor

ANN_DIR = os.environ.get(
    "SPLIT_ANN_DIR",
    "/home/ubuntu/us-west-3-fs/live_dealer_blackjack/action_annotation/split")
DET_DIR = os.environ.get(
    "DET_DIR",
    "/home/ubuntu/us-west-3-fs/sahithi/hard_action_annotations/card_detections_roundwise")
ROOTS = [
    "/home/ubuntu/us-west-3-fs/live_dealer_blackjack/roundcut/good_quality_rounds",
    "/home/ubuntu/us-west-3-fs/live_dealer_blackjack/roundcut/multidealer_300rounds",
    "/home/ubuntu/us-west-3-fs/live_dealer_blackjack/roundcut/remaining_vid",
    "/home/ubuntu/us-west-3-fs/live_dealer_blackjack/roundcut/good_quality_rounds_sparse_action_retrieved",
]
OUT = os.environ.get("SPLIT_OUT", "/home/ubuntu/sahithi/split_sam_output/clustered")
TRAJ_DIR = os.path.join(OUT, "trajectories")
VIZ_DIR = os.path.join(OUT, "viz")
os.makedirs(TRAJ_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)

N = int(sys.argv[1]) if len(sys.argv) > 1 else None
START = int(sys.argv[2]) if len(sys.argv) > 2 else 0
DO_VIZ = os.environ.get("VIZ", "1") != "0"
SAME_RANK_SUIT = os.environ.get("SAME_RANK_SUIT", "0") == "1"

DET_W, DET_H = 1920, 1080          # detection coordinate space
FACE = {"K", "Q", "J", "10"}
PLAYER_ZONE_Y = 450                # det space: cards below this are in the player area
PAIR_MIN_SEP = 25                  # det px: seeds must be distinct (avoid one-card collapse)
PAIR_MAX_SEP = 175                 # det px: a single player's split, not two distant seats
RANK_CONF_MIN = 0.5
NEAR_ANN_MAX = 300                 # det px: pair must be near the annotated split location
COLORS = [(0, 0, 255), (0, 220, 0)]


def norm_rank(r):
    if not r or r == "U":
        return None
    return "T" if r in FACE else r


def build_video_index():
    idx = {}
    for r in ROOTS:
        for p in glob.glob(os.path.join(r, "*.mp4")):
            idx.setdefault(os.path.basename(p), p)
    return idx


def seg_is_same_rank_suit(seg):
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
        v = d["video_name"]
        stem = os.path.splitext(v)[0]
        det = os.path.join(DET_DIR, f"{stem}_card.jsonl")
        if v not in idx or not os.path.exists(det):
            continue
        jobs.append((v, idx[v], det, segs))
    return jobs


def load_dets(det_path):
    """{frame_id: [ {c:(x,y) det-space, rank, nrank, suit, conf, rconf} ]}."""
    out = defaultdict(list)
    with open(det_path) as f:
        for line in f:
            o = json.loads(line)
            fr = o["frame_id"]
            for d in o.get("detections", []):
                c = d["polygon_center"]
                cx, cy = (c[0], c[1]) if isinstance(c[0], (int, float)) else (c[0][0], c[0][1])
                out[fr].append({
                    "c": (float(cx), float(cy)),
                    "rank": d.get("rank", ""), "nrank": norm_rank(d.get("rank", "")),
                    "suit": d.get("suit", ""),
                    "conf": float(d.get("conf", d.get("confidence", 0.0))),
                    "rconf": float(d.get("rank_conf", 0.0)),
                })
    return out


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def ann_anchor(seg):
    """(nrank, (cx,cy) in DET space) for the annotated pair, from a keyframe card_box."""
    for bb in seg.get("bounding_boxes", []):
        for kf in bb.get("keyframes", []):
            cb = kf.get("card_box")
            r = kf.get("rank") or []
            if cb and len(cb) >= 2 and r:
                cs = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in cb[:2]]
                mx = (cs[0][0] + cs[1][0]) / 2 * (DET_W / 1280.0)
                my = (cs[0][1] + cs[1][1]) / 2 * (DET_H / 720.0)
                return norm_rank(r[0]), (mx, my)
    return None, None


def find_pair(dets_by_frame, seg):
    """Scan the segment for the same-rank pair nearest the annotated location. Return
    the frame where that pair is MOST separated (distinct seeds), with the two centers."""
    nrank, anchor = ann_anchor(seg)
    if nrank is None:
        return None
    sf, ef = seg["start_frame"], seg["end_frame"]
    best = None  # (separation, frame, cardA, cardB)
    for fr in range(sf, ef + 1):
        cand = [d for d in dets_by_frame.get(fr, [])
                if d["nrank"] == nrank and d["rconf"] >= RANK_CONF_MIN
                and d["c"][1] >= PLAYER_ZONE_Y]
        for i in range(len(cand)):
            for j in range(i + 1, len(cand)):
                a, b = cand[i], cand[j]
                sep = _dist(a["c"], b["c"])
                if not (PAIR_MIN_SEP <= sep <= PAIR_MAX_SEP):
                    continue
                mid = ((a["c"][0] + b["c"][0]) / 2, (a["c"][1] + b["c"][1]) / 2)
                if _dist(mid, anchor) > NEAR_ANN_MAX:
                    continue
                if best is None or sep > best[0]:
                    best = (sep, fr, a, b)
    if best is None:
        return None
    _, fr, a, b = best
    # order by y (upper first) for stable card indexing
    if a["c"][1] > b["c"][1]:
        a, b = b, a
    return {"frame": fr, "rank": nrank, "cards": [a, b]}


def box_center_px(bx, W, H):
    x, y, w, h = [float(v) for v in bx]
    if max(x, y, w, h) <= 1.5:
        x, w = x * W, w * W
        y, h = y * H, h * H
    return x + w / 2, y + h / 2, (x, y, w, h)


def track_pair(predictor, sid, pair, W, H, num_frames, sf, ef):
    kf_frame = int(min(max(pair["frame"], 0), num_frames - 1))
    sx, sy = W / DET_W, H / DET_H
    centers = [(c["c"][0] * sx / W, c["c"][1] * sy / H) for c in pair["cards"]]  # normalized
    predictor.handle_request(dict(type="reset_session", session_id=sid))
    for oid, c in enumerate(centers):
        predictor.handle_request(dict(
            type="add_prompt", session_id=sid, frame_index=kf_frame,
            obj_id=oid, points=[list(c)], point_labels=[1], rel_coordinates=True))
    max_track = max(kf_frame - sf, ef - kf_frame) + 3
    per_card = defaultdict(list)
    boxes_by_frame = defaultdict(dict)
    max_area = defaultdict(float)
    for r in predictor.handle_stream_request(dict(
            type="propagate_in_video", session_id=sid, start_frame_index=kf_frame,
            max_frame_num_to_track=max_track, propagation_direction="both")):
        fi = r["frame_index"]
        if not (sf <= fi <= ef):
            continue
        o = r["outputs"]
        ids = o["out_obj_ids"]; ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        obx = o["out_boxes_xywh"]
        probs = o.get("out_probs")
        probs = probs.tolist() if hasattr(probs, "tolist") else (list(probs) if probs is not None else None)
        for j, oid in enumerate(ids):
            bx = obx[j]; bx = bx.tolist() if hasattr(bx, "tolist") else list(bx)
            cx, cy, xywh = box_center_px(bx, W, H)
            p = float(probs[j]) if probs is not None and j < len(probs) else None
            per_card[int(oid)].append([fi, round(cx, 1), round(cy, 1), round(p, 3) if p is not None else None])
            boxes_by_frame[fi][int(oid)] = xywh
            max_area[int(oid)] = max(max_area[int(oid)], xywh[2] * xywh[3])

    cards = []
    for oid in sorted(per_card):
        pts = sorted(per_card[oid], key=lambda t: t[0])
        traj, prev = [], None
        for (f, cx, cy, p) in pts:
            mv = 1 if prev and math.hypot(cx - prev[0], cy - prev[1]) > 2.5 else 0
            traj.append([f, cx, cy, mv, p]); prev = (cx, cy)
        src = pair["cards"][oid] if oid < len(pair["cards"]) else {}
        cards.append({
            "card_index": oid,
            "rank": src.get("rank"), "suit": src.get("suit"),
            "final_class": (str(src.get("rank")) + str(src.get("suit"))) if src.get("rank") and src.get("suit") else None,
            "seed_center_det": src.get("c"),
            "n_frames": len(traj),
            "max_box_frac": round(max_area[oid] / (W * H), 4),
            "trajectory": traj,
        })
    return {"prompt_frame": kf_frame, "cards": cards}, boxes_by_frame


def render_viz(video_path, rec, boxes_by_frame, out_path):
    sf, ef = rec["start_frame"], rec["end_frame"]
    cap = cv2.VideoCapture(video_path)
    W = int(cap.get(3)); H = int(cap.get(4)); fps = cap.get(5) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, sf)
    tmp = out_path + ".tmp.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    trails = defaultdict(list)
    for f in range(sf, ef + 1):
        ret, img = cap.read()
        if not ret:
            break
        for card in rec["cards"]:
            for (ff, cx, cy, mv, p) in card["trajectory"]:
                if ff == f:
                    trails[card["card_index"]].append((int(cx), int(cy)))
        for card in rec["cards"]:
            oid = card["card_index"]; color = COLORS[oid % len(COLORS)]
            xywh = boxes_by_frame.get(f, {}).get(oid)
            if xywh:
                x, y, w, h = xywh
                cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
                cv2.putText(img, f"card{oid} {card['final_class']}", (int(x), max(0, int(y) - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            for k in range(1, len(trails[oid])):
                cv2.line(img, trails[oid][k - 1], trails[oid][k], color, 2)
            if trails[oid]:
                cv2.circle(img, trails[oid][-1], 3, color, -1)
        cv2.putText(img, f"f{f} [{sf}-{ef}] split {rec.get('rank','')}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        vw.write(img)
    vw.release(); cap.release()
    rc = os.system(f'ffmpeg -y -loglevel error -i "{tmp}" -c:v libx264 -pix_fmt yuv420p '
                   f'-movflags +faststart "{out_path}" 2>/dev/null')
    if rc == 0 and os.path.exists(out_path):
        os.remove(tmp)
    else:
        os.replace(tmp, out_path)


def main():
    idx = build_video_index()
    jobs = load_jobs(idx)
    print(f"[info] {len(jobs)} split videos with detections", flush=True)
    jobs = jobs[START:(START + N) if N else None]
    print(f"[info] processing {len(jobs)} (START={START} N={N}) VIZ={DO_VIZ} SRS={SAME_RANK_SUIT}", flush=True)

    predictor = build_sam3_video_predictor()
    stats = {"videos": 0, "segments": 0, "tracked": 0, "no_pair": 0, "failed": []}
    for vi, (vname, vpath, det_path, segs) in enumerate(jobs):
        stem = os.path.splitext(vname)[0]
        traj_path = os.path.join(TRAJ_DIR, f"{stem}_split_trajectory.jsonl")
        if os.path.exists(traj_path):
            print(f"[skip] {stem}", flush=True); continue
        try:
            dets_by_frame = load_dets(det_path)
            cap = cv2.VideoCapture(vpath)
            W = int(cap.get(3)); H = int(cap.get(4)); num_frames = int(cap.get(7)); cap.release()
            sid = predictor.handle_request(dict(type="start_session", resource_path=vpath))["session_id"]
            lines = []
            for k, seg in enumerate(segs):
                stats["segments"] += 1
                pair = find_pair(dets_by_frame, seg)
                if pair is None:
                    stats["no_pair"] += 1
                    print(f"   [warn] {stem} seg{k}: no same-rank pair in detections", flush=True)
                    continue
                sf = int(max(seg["start_frame"], 0)); ef = int(min(seg["end_frame"], num_frames - 1))
                trk, boxes_by_frame = track_pair(predictor, sid, pair, W, H, num_frames, sf, ef)
                rec = {"video": stem, "segment_id": seg.get("id"), "action": "split",
                       "rank": pair["rank"], "start_frame": sf, "end_frame": ef,
                       "prompt_frame": trk["prompt_frame"], "resolution": f"{W}x{H}",
                       "n_cards": len(trk["cards"]), "cards": trk["cards"]}
                lines.append(rec); stats["tracked"] += 1
                if DO_VIZ:
                    render_viz(vpath, rec, boxes_by_frame, os.path.join(VIZ_DIR, f"{stem}_seg{k}.mp4"))
            predictor.handle_request(dict(type="close_session", session_id=sid))
            with open(traj_path, "w") as fh:
                for rec in lines:
                    fh.write(json.dumps(rec) + "\n")
            stats["videos"] += 1
            print(f"[{vi+1}/{len(jobs)}] {stem}: {len(lines)} tracked -> {traj_path}", flush=True)
        except Exception as e:
            stats["failed"].append({"video": stem, "error": str(e)})
            print(f"[FAIL] {stem}: {e}", flush=True); traceback.print_exc()
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
