"""Interactive web UI to step a round's deal forward.

Loads a per_card.json + _vis.jpg, treats the first 2 cards per seat as the
initial deal, and lets the user click Hit / Double per seat to extend the
state with predicted card placements (powered by predict_next_card.py).

Run:
    python web_app.py
Then open http://<host>:5050/
"""

import io
import json
import math
import threading
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template_string, request, send_file

from intra_pile_util import point_in_obb
from predict_next_card import (
    DEFAULT_CARDS_DIR,
    DOUBLE_ANGLE_DEG,
    HIT_ANGLE_DEG,
    STEP_MAGNITUDE_PX,
    _rotate,
)

OUT_DIR = "/home/ubuntu/us-west-3-fs/sahithi/clus_anno/intra_pile_v1_double_only_1k"
SPLIT_BLOCKLIST_PATH = "/home/ubuntu/us-west-3-fs/sahithi/clus_anno/intra_pile_v1_split_rounds.txt"
POLYDRAW_PATH = "/home/ubuntu/us-west-3-fs/sahithi/clus_anno/polydraw_json/polydraw-merged.json"
DEALER_SEAT_ID = 0
VIDEO_DIR = "/home/ubuntu/us-west-3-fs/live_dealer_blackjack/roundcut/good_quality_rounds"
FIRST_FRAME_CACHE = Path("/tmp/first_frames")
FIRST_FRAME_CACHE.mkdir(exist_ok=True)
_polydraw_cache = None
_polydraw_center_cache = {}


def _load_polydraw():
    global _polydraw_cache, _polydraw_center_cache
    if _polydraw_cache is not None:
        return
    with open(POLYDRAW_PATH) as f:
        d = json.load(f)
    by_id = {p["id"]: (p["x"], p["y"]) for p in d["points"]}
    polys, centers = {}, {}
    for g in d["pointGroups"]:
        name = g["name"]
        if name == "dealer":
            sid = DEALER_SEAT_ID
        elif name.startswith("p") and name[1:].isdigit():
            sid = int(name[1:])
        else:
            continue
        corners_px = [by_id[i] for i in g["pointIds"]]
        cx = sum(c[0] for c in corners_px) / 4
        cy = sum(c[1] for c in corners_px) / 4
        offs = [(c[0] - cx, c[1] - cy) for c in corners_px]
        offs.sort(key=lambda p: math.atan2(p[1], p[0]))
        polys[sid] = offs
        centers[sid] = (cx, cy)
    _polydraw_cache = polys
    _polydraw_center_cache = centers


def _polygon_for_seat(seat_id):
    _load_polydraw()
    return _polydraw_cache.get(seat_id, [(-30, -22), (30, -22), (30, 22), (-30, 22)])


def _polygon_center_for_seat(seat_id):
    _load_polydraw()
    return _polydraw_center_cache.get(seat_id, (640.0, 360.0))


def _split_blocklist():
    try:
        return set(open(SPLIT_BLOCKLIST_PATH).read().splitlines())
    except FileNotFoundError:
        return set()

app = Flask(__name__)
sessions = {}  # round_name -> {seat_id: [{"center","bbox","is_double"}, ...]}
_round_cache = {}  # round_name -> per_card.json data
_bbox_cache = {}  # (round_name, seat_id) -> (big_w, big_h)
_rep_frame_cache = {}  # round_name -> rep_fid frame dict
_initial_cache = {}  # (round_name, seat_id) -> [(cx, cy), ...]
placed_initial = {}  # round_name -> True once "Initial hand" was clicked


def _round_data(name):
    if name in _round_cache:
        return _round_cache[name]
    p = Path(OUT_DIR) / f"{name}_per_card.json"
    if not p.exists():
        return None
    data = json.load(open(p))
    _round_cache[name] = data
    return data


def _seats_present(data):
    """Seat ids (1..7) that have at least one ordinal-tagged track in this
    round — i.e., players actually dealt to in the source video — plus the
    dealer (seat 0) if a dealer polygon is annotated."""
    seats = set()
    for t in data.get("tracks", []):
        sid = t.get("seat")
        if sid in (None, 0):
            continue
        if t.get("ordinal") is not None:
            seats.add(sid)
    out = sorted(seats)
    _load_polydraw()
    if DEALER_SEAT_ID in _polydraw_cache:
        out.append(DEALER_SEAT_ID)
    return out


def _deal_order_for(data):
    """Deal order: pass 1 = P1#1..P7#1 then D#1, pass 2 = P1#2..P7#2 then D#2,
    skipping seats not present in this round."""
    seats = _seats_present(data)
    players = [s for s in seats if s != DEALER_SEAT_ID]
    has_dealer = DEALER_SEAT_ID in seats
    order = []
    for idx in (0, 1):
        for sid in players:
            order.append((sid, idx))
        if has_dealer:
            order.append((DEALER_SEAT_ID, idx))
    return order


def _rep_frame(round_name, data):
    if round_name in _rep_frame_cache:
        return _rep_frame_cache[round_name]
    rep_fid = int(data["representative_frame"].split("=")[-1])
    jsonl = Path(DEFAULT_CARDS_DIR) / f"{data['round']}_card.jsonl"
    found = None
    with open(jsonl) as f:
        for line in f:
            fr = json.loads(line)
            if fr.get("frame_id") == rep_fid:
                found = fr
                break
    _rep_frame_cache[round_name] = found
    return found


def _seat_dets_at_rep(round_name, data, seat_id):
    rep = _rep_frame(round_name, data)
    if rep is None:
        return []
    corners = data["seat_obbs"].get(str(seat_id)) or data["seat_obbs"].get(seat_id)
    if corners is None:
        return []
    out = []
    for det in rep.get("detections") or []:
        pc = det.get("polygon_center")
        if pc is None:
            continue
        if isinstance(pc[0], list):
            pc = pc[0]
        if (det.get("rank"), det.get("suit")) == ("CB", "CB"):
            continue
        if point_in_obb(pc, corners):
            out.append({"rank": det.get("rank"), "suit": det.get("suit"),
                        "center": tuple(pc), "box": det["box"]})
    return out


def _raw_initial_centers(data, seat_id, round_name):
    """First 2 rep_fid centroids matched to this seat's ord-1/ord-2 tracks by
    rank+suit. Falls back to the track's last_center when no rep_fid match
    is found. May return fewer than 2 centers if tracks are missing."""
    base = [t for t in data["tracks"]
            if t.get("seat") == seat_id and t.get("ordinal") is not None]
    base.sort(key=lambda t: t["ordinal"])
    base = base[:2]
    rep_dets = _seat_dets_at_rep(round_name, data, seat_id)
    out = []
    for t in base:
        match, best_d = None, math.inf
        lc = t["last_center"]
        for det in rep_dets:
            if (det["rank"], det["suit"]) != (t["rank"], t["suit"]):
                continue
            d = (det["center"][0] - lc[0]) ** 2 + (det["center"][1] - lc[1]) ** 2
            if d < best_d:
                best_d, match = d, det
        out.append(tuple(match["center"]) if match else tuple(lc))
    return out


_stat_initial_cache = None
_stat_initial_lock = threading.Lock()
STAT_CACHE_FILE = Path("/tmp/stat_initial_cache.json")


def _raw_initial_centers_fast(data, seat_id):
    """Last-center of this seat's ord-1/ord-2 tracks. Cheap version of
    _raw_initial_centers that skips the rep-frame jsonl scan — used for
    statistics aggregation across many rounds where matching to rep_fid
    detections would be prohibitively slow."""
    base = [t for t in data["tracks"]
            if t.get("seat") == seat_id and t.get("ordinal") is not None]
    base.sort(key=lambda t: t["ordinal"])
    return [tuple(t["last_center"]) for t in base[:2]]


def _statistical_initial_centers():
    """Population median position per (seat, ord_idx) across all listed
    rounds, using only centers that fall inside the seat OBB so noisy
    matches don't pollute the median. ord_idx 0 = ord-1, 1 = ord-2.
    Guarded by a lock so concurrent requests don't all rebuild it at once."""
    global _stat_initial_cache
    if _stat_initial_cache is not None:
        return _stat_initial_cache
    with _stat_initial_lock:
        if _stat_initial_cache is not None:
            return _stat_initial_cache
        if STAT_CACHE_FILE.exists():
            try:
                raw = json.loads(STAT_CACHE_FILE.read_text())
                _stat_initial_cache = {
                    tuple(int(x) for x in k.split(",")): tuple(v) for k, v in raw.items()
                }
                return _stat_initial_cache
            except Exception:
                pass
        seat_ids = list(range(1, 8)) + [DEALER_SEAT_ID]
        pts = {(sid, idx): [] for sid in seat_ids for idx in (0, 1)}
        for r in _list_rounds():
            data = _round_data(r["name"])
            if data is None:
                continue
            for sid in seat_ids:
                corners = data["seat_obbs"].get(str(sid)) or data["seat_obbs"].get(sid)
                for idx, c in enumerate(_raw_initial_centers_fast(data, sid)):
                    if corners and point_in_obb(c, corners):
                        pts[(sid, idx)].append(c)
        result = {}
        for key, vs in pts.items():
            if vs:
                xs = sorted(p[0] for p in vs)
                ys = sorted(p[1] for p in vs)
                mid = len(vs) // 2
                result[key] = (xs[mid], ys[mid])
        _stat_initial_cache = result
        try:
            STAT_CACHE_FILE.write_text(
                json.dumps({f"{k[0]},{k[1]}": list(v) for k, v in result.items()}))
        except Exception:
            pass
        return result


def _dealer_step_px():
    """Side-by-side horizontal step between dealer cards = dealer polygon's
    local x-extent, so adjacent dealer cards just touch."""
    poly = _polygon_for_seat(DEALER_SEAT_ID)
    xs = [p[0] for p in poly]
    return max(xs) - min(xs)


def _dealer_initial_centers():
    """Two dealer cards laid side by side, centered on the dealer's
    statistical card position (median of seat-0 tracks across rounds, same
    source players use). Falls back to the dealer polygon center if no
    statistics are available."""
    stats = _statistical_initial_centers()
    pts = [stats[(DEALER_SEAT_ID, i)] for i in (0, 1)
           if (DEALER_SEAT_ID, i) in stats]
    if pts:
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
    else:
        cx, cy = _polygon_center_for_seat(DEALER_SEAT_ID)
    half = _dealer_step_px() / 2.0
    return [(cx - half, cy), (cx + half, cy)]


def _initial_centers(data, seat_id, round_name=None):
    """For players: statistical median position per (seat, ord_idx) across
    all rounds. For dealer (seat 0): fixed side-by-side around the dealer
    polygon center. data/round_name accepted for caller convenience."""
    if seat_id == DEALER_SEAT_ID:
        return _dealer_initial_centers()
    stats = _statistical_initial_centers()
    out = []
    for idx in range(2):
        if (seat_id, idx) in stats:
            out.append(stats[(seat_id, idx)])
    return out


def _build_polygon_at(seat_id, center, rotation_deg=0.0):
    poly_local = _polygon_for_seat(seat_id)
    if rotation_deg:
        c = math.cos(math.radians(rotation_deg))
        s = math.sin(math.radians(rotation_deg))
        poly_local = [(p[0] * c + p[1] * s, -p[0] * s + p[1] * c)
                      for p in poly_local]
    return [[center[0] + p[0], center[1] + p[1]] for p in poly_local]


def _poly_bbox(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return (min(xs), min(ys), max(xs), max(ys))


def _inflate_bbox(b, m):
    """Grow a bbox by margin m on every side."""
    return (b[0] - m, b[1] - m, b[2] + m, b[3] + m)


def _bbox_overlap_ratio(a, b):
    """max(intersection_area / area_a, intersection_area / area_b)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max((a[2] - a[0]) * (a[3] - a[1]), 1e-6)
    area_b = max((b[2] - b[0]) * (b[3] - b[1]), 1e-6)
    return max(inter / area_a, inter / area_b)


def _seat_centers(data, seat_id, extras):
    round_name = data["round"]
    centers = (list(_initial_centers(data, seat_id))
               if placed_initial.get(round_name)
                  and seat_id in _seats_present(data)
               else [])
    for add in extras.get(seat_id, []):
        centers.append(tuple(add["center"]))
    return centers


def _all_centers(data, extras):
    out = []
    for seat_id in range(1, 8):
        out.extend(_initial_centers(data, seat_id))
    out.extend(tuple(t["last_center"]) for t in data["tracks"]
               if t.get("seat") == 0 and t.get("ordinal") is not None)
    for sid, adds in extras.items():
        for add in adds:
            out.append(tuple(add["center"]))
    return out


def _bigger_bbox(round_name, data, seat_id):
    key = (round_name, seat_id)
    if key in _bbox_cache:
        return _bbox_cache[key]
    dets = _seat_dets_at_rep(round_name, data, seat_id)
    if dets:
        big = (max(d["box"][2] - d["box"][0] for d in dets),
               max(d["box"][3] - d["box"][1] for d in dets))
    else:
        big = (40.0, 60.0)
    _bbox_cache[key] = big
    return big


CLUSTER_DISPLACE_STEP_PX = 4.0
CLUSTER_DISPLACE_MAX_PX = 80.0
OVERLAP_RATIO_THRESHOLD = 0.1
REBALANCE_MAX_PASSES = 5

MAX_CLUSTER_DY_PX = 120  # |chain y extent from card2| cap (toward dealer)
MIN_ABS_Y_PX = 470  # no card center may have y below this (image-absolute)
MIN_SHRINK_RATIO = 0.15  # don't shrink below this to keep cards distinguishable
DEALER_CLEARANCE_PX = 4.0  # margin enforced between any player card and dealer cards
SHRINK_SEARCH_STEP = 0.05  # ratio decrement when searching for dealer clearance

seat_displacements = {}  # (round_name, seat_id) -> (dx, dy) shift in px


def _displaced_initial(round_name, data, seat_id):
    """Statistical initial centers for a seat plus any persisted cluster
    displacement applied to keep the whole cluster (initial + extras) shifted
    together away from inter-seat collisions. Dealer (seat 0) is never
    displaced: its cards are pinned to the dealer polygon."""
    centers = _initial_centers(data, seat_id)
    if seat_id == DEALER_SEAT_ID:
        return list(centers)
    disp = seat_displacements.get((round_name, seat_id), (0.0, 0.0))
    return [(c[0] + disp[0], c[1] + disp[1]) for c in centers]


def _cluster_shrink_ratio(positions, card2):
    """Find the largest ratio in (MIN_SHRINK_RATIO, 1] that, applied uniformly
    to each position's offset from card2, keeps the chain within
    MAX_CLUSTER_DY_PX of card2 in y AND no card crossing above MIN_ABS_Y_PX
    in the image (i.e., card.y stays >= MIN_ABS_Y_PX). Applied uniformly so
    the whole cluster contracts together rather than only the last card."""
    ratio = 1.0
    for p in positions:
        dy = p[1] - card2[1]
        if abs(dy) > MAX_CLUSTER_DY_PX:
            need_dy = MAX_CLUSTER_DY_PX / abs(dy)
            if need_dy < ratio:
                ratio = need_dy
        if p[1] < MIN_ABS_Y_PX and dy < 0:
            # card2.y + ratio*dy >= MIN_ABS_Y_PX  =>  ratio <= (MIN_ABS_Y_PX - card2.y) / dy
            need_abs = (MIN_ABS_Y_PX - card2[1]) / dy
            if 0 < need_abs < ratio:
                ratio = need_abs
    return max(ratio, MIN_SHRINK_RATIO)


def _dealer_clear_ratio(seat_id, card2, steps, disp, dealer_bboxes, start_ratio):
    """Largest ratio <= start_ratio (down to MIN_SHRINK_RATIO) such that the
    player's shrunk hit-chain — card2 + cumulative ratio*step + disp — has no
    card bbox overlapping any dealer card bbox. Player cards must never
    collide with the dealer, so when the chain reaches the dealer's cards the
    player side contracts. Returns MIN_SHRINK_RATIO if no ratio clears."""
    if not dealer_bboxes:
        return start_ratio
    ratio = start_ratio
    while ratio >= MIN_SHRINK_RATIO:
        collide = False
        cur = card2
        for s in steps:
            cur = (cur[0] + s[0] * ratio, cur[1] + s[1] * ratio)
            pb = _poly_bbox(_build_polygon_at(seat_id, (cur[0] + disp[0], cur[1] + disp[1])))
            if any(_bbox_overlap_ratio(pb, db) > 0.0 for db in dealer_bboxes):
                collide = True
                break
        if not collide:
            return ratio
        ratio -= SHRINK_SEARCH_STEP
    return MIN_SHRINK_RATIO


def _rebalance_seat(round_name, seat_id):
    """Displace seat_id's session cluster opposite to the colliding seat's
    cluster centroid to keep their bbox overlap below threshold. Persists
    the accumulated displacement across calls so a shifted cluster doesn't
    snap back. Dealer (seat 0) is never rebalanced — its cards are pinned
    to the dealer polygon and laid out side by side."""
    if seat_id == DEALER_SEAT_ID:
        return False
    data = _round_data(round_name)
    extras = sessions.setdefault(round_name, {})
    existing = extras.get(seat_id, [])
    if not existing:
        return False
    initial = _initial_centers(data, seat_id)
    if len(initial) < 2:
        return False
    card1, card2 = tuple(initial[0]), tuple(initial[1])
    v0 = (card2[0] - card1[0], card2[1] - card1[1])
    mag = math.hypot(*v0)
    base_step = ((v0[0] * STEP_MAGNITUDE_PX / mag,
                  v0[1] * STEP_MAGNITUDE_PX / mag)
                 if mag > 1e-6 else (0.0, -STEP_MAGNITUDE_PX))
    all_double = [a["is_double"] for a in existing]

    other_bboxes = []
    for sid in range(1, 8):
        if sid == seat_id:
            continue
        for c in _displaced_initial(round_name, data, sid):
            other_bboxes.append((_poly_bbox(_build_polygon_at(sid, c)), c))
        for add in extras.get(sid, []):
            other_bboxes.append((_poly_bbox(add["polygon"]), tuple(add["center"])))

    dealer_bboxes = []
    for c in _displaced_initial(round_name, data, DEALER_SEAT_ID):
        dealer_bboxes.append(_inflate_bbox(_poly_bbox(_build_polygon_at(DEALER_SEAT_ID, c)),
                                           DEALER_CLEARANCE_PX))
    for add in extras.get(DEALER_SEAT_ID, []):
        dealer_bboxes.append(_inflate_bbox(_poly_bbox(add["polygon"]), DEALER_CLEARANCE_PX))

    def compute(disp):
        initial_pos = [(c[0] + disp[0], c[1] + disp[1]) for c in initial]
        steps = [_rotate(base_step, DOUBLE_ANGLE_DEG if d else HIT_ANGLE_DEG)
                 for d in all_double]
        unc, cur = [], card2
        for s in steps:
            cur = (cur[0] + s[0], cur[1] + s[1])
            unc.append(cur)
        ratio = _cluster_shrink_ratio(unc, card2)
        ratio = _dealer_clear_ratio(seat_id, card2, steps, disp, dealer_bboxes, ratio)
        extras_pos, cur = [], card2
        for s in steps:
            cur = (cur[0] + s[0] * ratio, cur[1] + s[1] * ratio)
            extras_pos.append((cur[0] + disp[0], cur[1] + disp[1]))
        return initial_pos, extras_pos

    def worst(initial_pos, extras_pos):
        worst_r, worst_off = 0.0, None
        for pos in initial_pos + extras_pos:
            pb = _poly_bbox(_build_polygon_at(seat_id, pos))
            for ob, oc in other_bboxes:
                r = _bbox_overlap_ratio(pb, ob)
                if r > worst_r:
                    worst_r, worst_off = r, oc
        return worst_r, worst_off

    disp = seat_displacements.get((round_name, seat_id), (0.0, 0.0))
    start_disp = disp
    last_dir = (0.0, 0.0)
    while math.hypot(*disp) <= CLUSTER_DISPLACE_MAX_PX:
        initial_pos, extras_pos = compute(disp)
        ratio, offender = worst(initial_pos, extras_pos)
        if ratio <= OVERLAP_RATIO_THRESHOLD or offender is None:
            break
        all_pos = initial_pos + extras_pos
        cx = sum(p[0] for p in all_pos) / len(all_pos)
        vx = cx - offender[0]
        if abs(vx) < 1e-6:
            break
        step = ((1.0 if vx > 0 else -1.0) * CLUSTER_DISPLACE_STEP_PX, 0.0)
        if last_dir != (0.0, 0.0) and (step[0] * last_dir[0]
                                        + step[1] * last_dir[1]) < 0:
            break
        disp = (disp[0] + step[0], disp[1] + step[1])
        last_dir = step

    seat_displacements[(round_name, seat_id)] = disp
    _, extras_pos = compute(disp)
    polys = [_build_polygon_at(seat_id, p) for p in extras_pos]
    new_extras = [
        {"center": list(extras_pos[i]), "polygon": polys[i], "is_double": d}
        for i, d in enumerate(all_double)
    ]
    changed = disp != start_disp or new_extras != existing
    if changed:
        extras[seat_id] = new_extras
    return changed


def _rebalance_all(round_name):
    for _ in range(REBALANCE_MAX_PASSES):
        changed_any = False
        for sid in range(1, 8):
            if _rebalance_seat(round_name, sid):
                changed_any = True
        if not changed_any:
            return


def _predict_for_seat(round_name, seat_id, is_double):
    data = _round_data(round_name)
    if data is None:
        raise ValueError(f"round not found: {round_name}")
    extras = sessions.setdefault(round_name, {})
    label = "Dealer" if seat_id == DEALER_SEAT_ID else f"P{seat_id}"
    n_before = len(_seat_centers(data, seat_id, extras))
    if n_before < 2:
        raise ValueError(f"{label} needs the initial 2-card deal first (have {n_before})")
    if is_double:
        if seat_id == DEALER_SEAT_ID:
            raise ValueError("dealer cannot double")
        if n_before != 2:
            raise ValueError(f"double requires exactly 2 cards at {label}; have {n_before}")

    initial = _initial_centers(data, seat_id)
    if len(initial) < 2:
        raise ValueError(f"{label} initial deal missing")
    existing = extras.get(seat_id, [])
    last_pos = existing[-1]["center"] if existing else tuple(initial[1])

    if seat_id == DEALER_SEAT_ID:
        step = _dealer_step_px()
        new_pos = (last_pos[0] + step, last_pos[1])
    else:
        card1, card2 = tuple(initial[0]), tuple(initial[1])
        v0 = (card2[0] - card1[0], card2[1] - card1[1])
        mag = math.hypot(*v0)
        base_step = ((v0[0] * STEP_MAGNITUDE_PX / mag,
                      v0[1] * STEP_MAGNITUDE_PX / mag)
                     if mag > 1e-6 else (0.0, -STEP_MAGNITUDE_PX))
        base_ang = DOUBLE_ANGLE_DEG if is_double else HIT_ANGLE_DEG
        step_rot = _rotate(base_step, base_ang)
        new_pos = (last_pos[0] + step_rot[0], last_pos[1] + step_rot[1])

    extras.setdefault(seat_id, []).append({
        "center": list(new_pos),
        "polygon": _build_polygon_at(seat_id, new_pos),
        "is_double": is_double,
    })

    # Always rebalance: a dealer hit can newly intrude on a player's chain,
    # and players must shrink to clear the dealer. _rebalance_seat is a no-op
    # for the dealer itself (its cards stay pinned).
    _rebalance_all(round_name)

    final = extras[seat_id][-1]
    return {
        "center": final["center"],
        "polygon": final["polygon"],
        "chosen_angle_deg": 0.0,
    }


def _first_frame(round_name):
    cached = FIRST_FRAME_CACHE / f"{round_name}.png"
    if cached.exists():
        return str(cached)
    video = Path(VIDEO_DIR) / f"{round_name}.mp4"
    if not video.exists():
        return None
    cap = cv2.VideoCapture(str(video))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    if (frame.shape[1], frame.shape[0]) != (1280, 720):
        frame = cv2.resize(frame, (1280, 720))
    cv2.imwrite(str(cached), frame)
    return str(cached)


def _render_round(round_name):
    bg_path = _first_frame(round_name)
    img = cv2.imread(bg_path) if bg_path else None
    if img is None:
        return None
    data = _round_data(round_name)
    if data is not None and placed_initial.get(round_name):
        for sid in _seats_present(data):
            for idx, c in enumerate(_displaced_initial(round_name, data, sid)):
                poly = _build_polygon_at(sid, c)
                pts = np.array([[int(p[0]), int(p[1])] for p in poly],
                               dtype=np.int32).reshape(-1, 1, 2)
                col = CARD1_COLOR_BGR if idx == 0 else CARD2_COLOR_BGR
                cv2.polylines(img, [pts], True, (0, 0, 0), 5)
                cv2.polylines(img, [pts], True, col, 2)
                cv2.circle(img, (int(c[0]), int(c[1])), 3, col, -1)
    extras = sessions.get(round_name, {})
    for sid, adds in extras.items():
        for idx, add in enumerate(adds):
            col = (0, 165, 255) if add["is_double"] else (255, 255, 255)
            pts = np.array([[int(p[0]), int(p[1])] for p in add["polygon"]],
                           dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], True, (0, 0, 0), 5)
            cv2.polylines(img, [pts], True, col, 3)
            cx, cy = [int(v) for v in add["center"]]
            cv2.circle(img, (cx, cy), 4, col, -1)
            prefix = "D" if sid == DEALER_SEAT_ID else f"P{sid}"
            tag = f"{prefix}+{idx + 1}{'D' if add['is_double'] else 'H'}"
            x0 = min(p[0] for p in add["polygon"])
            y0 = min(p[1] for p in add["polygon"])
            cv2.putText(img, tag, (int(x0), int(y0) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
            cv2.putText(img, tag, (int(x0), int(y0) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes() if ok else None


_list_rounds_cache = None


def _list_rounds():
    global _list_rounds_cache
    if _list_rounds_cache is not None:
        return _list_rounds_cache
    blocklist = _split_blocklist()
    rounds = []
    for p in sorted(Path(OUT_DIR).glob("*_per_card.json")):
        if p.name.startswith("_"):
            continue
        try:
            data = json.load(open(p))
        except Exception:
            continue
        if data["round"] in blocklist:
            continue
        counts = {}
        for t in data["tracks"]:
            sid = t.get("seat")
            if sid is None or sid == 0:
                continue
            counts[sid] = counts.get(sid, 0) + 1
        if any(v >= 2 for v in counts.values()):
            counts_str = " ".join(f"P{k}:{counts[k]}" for k in sorted(counts))
            rounds.append({"name": data["round"], "counts": counts_str})
    _list_rounds_cache = rounds
    return rounds


def _seat_states(round_name):
    data = _round_data(round_name)
    extras = sessions.setdefault(round_name, {})
    seats = {}
    for sid in list(range(1, 8)) + [DEALER_SEAT_ID]:
        n = len(_seat_centers(data, sid, extras))
        is_dealer = sid == DEALER_SEAT_ID
        seats[sid] = {
            "n_cards": n,
            "can_double": (not is_dealer) and n == 2,
            "can_hit": n >= 2,
        }
    present = _seats_present(data) if data is not None else []
    return {"seats": seats, "present_seats": present,
            "initial_hand_steps": len(present) * 2}


@app.route("/")
def index():
    return render_template_string(INDEX_HTML, rounds=_list_rounds())


@app.route("/round/<name>")
def round_state(name):
    data = _round_data(name)
    if data is None:
        return jsonify({"error": "round not found"}), 404
    return jsonify({"round": name, **_seat_states(name)})


@app.route("/predict/<name>", methods=["POST"])
def predict(name):
    body = request.get_json() or {}
    try:
        sid = int(body["seat"])
        is_double = body.get("action") == "double"
        result = _predict_for_seat(name, sid, is_double)
    except (KeyError, ValueError) as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"round": name, **_seat_states(name),
                    "last_action": {"seat": sid, "is_double": is_double, **result}})


@app.route("/reset/<name>", methods=["POST"])
def reset(name):
    sessions[name] = {}
    placed_initial.pop(name, None)
    for k in [k for k in seat_displacements if k[0] == name]:
        del seat_displacements[k]
    return jsonify({"round": name, **_seat_states(name)})


@app.route("/place_initial/<name>", methods=["POST"])
def place_initial(name):
    if _round_data(name) is None:
        return jsonify({"error": "round not found"}), 404
    placed_initial[name] = True
    return jsonify({"round": name, **_seat_states(name)})


SAVE_BASE_DIR = Path("/home/ubuntu/us-west-3-fs/sahithi/clus_anno/card_placement_with_dealer")
SAVE_JSON_DIR = SAVE_BASE_DIR / "json"
SAVE_IMAGE_ORIGINAL_DIR = SAVE_BASE_DIR / "image_original"
SAVE_OVERLAY_DIR = SAVE_BASE_DIR / "overlay_image"


def _bbox_from_poly(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]


@app.route("/save/<name>", methods=["POST"])
def save_placement(name):
    data = _round_data(name)
    if data is None:
        return jsonify({"error": "round not found"}), 404
    if not placed_initial.get(name):
        return jsonify({"error": "click 'Initial hand' before saving"}), 400
    SAVE_JSON_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_IMAGE_ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    extras = sessions.get(name, {})
    out = {"round": name, "present_seats": _seats_present(data), "seats": {}}
    if placed_initial.get(name):
        for sid in _seats_present(data):
            cards = []
            for idx, c in enumerate(_displaced_initial(name, data, sid)):
                poly = _build_polygon_at(sid, c)
                cards.append({
                    "ordinal": idx + 1,
                    "kind": "initial",
                    "center": [float(c[0]), float(c[1])],
                    "polygon": [[float(p[0]), float(p[1])] for p in poly],
                    "bbox": _bbox_from_poly(poly),
                })
            for j, add in enumerate(extras.get(sid, [])):
                cards.append({
                    "ordinal": 2 + j + 1,
                    "kind": "double" if add["is_double"] else "hit",
                    "center": [float(add["center"][0]), float(add["center"][1])],
                    "polygon": [[float(p[0]), float(p[1])] for p in add["polygon"]],
                    "bbox": _bbox_from_poly(add["polygon"]),
                })
            out["seats"][str(sid)] = {"cards": cards}
    json_path = SAVE_JSON_DIR / f"{name}.json"
    json_path.write_text(json.dumps(out, indent=2))
    original_path = SAVE_IMAGE_ORIGINAL_DIR / f"{name}.png"
    bg_path = _first_frame(name)
    if bg_path:
        original_path.write_bytes(Path(bg_path).read_bytes())
    overlay_path = SAVE_OVERLAY_DIR / f"{name}.png"
    img_bytes = _render_round(name)
    if img_bytes:
        overlay_path.write_bytes(img_bytes)
    return jsonify({"ok": True, "json": str(json_path),
                    "image_original": str(original_path) if bg_path else None,
                    "overlay_image": str(overlay_path) if img_bytes else None,
                    "seats_saved": len(out["seats"])})


@app.route("/vis/<name>.png")
def vis(name):
    img_bytes = _render_round(name)
    if img_bytes is None:
        return "not found", 404
    return send_file(io.BytesIO(img_bytes), mimetype="image/png")


CARD1_COLOR_BGR = (60, 60, 220)    # red-ish: ord-1 (first card)
CARD2_COLOR_BGR = (220, 200, 60)   # cyan-ish: ord-2 (second card)
_heatmap_cache = {}  # (frozenset(round_names), seat_filter) -> png bytes
HEATMAP_DISK_CACHE = Path("/tmp/heatmap_cache")
HEATMAP_DISK_CACHE.mkdir(exist_ok=True)


def _heatmap_image(seat_filter=None):
    import hashlib
    rounds = _list_rounds()
    if not rounds:
        return None
    key = (frozenset(r["name"] for r in rounds), seat_filter)
    if key in _heatmap_cache:
        return _heatmap_cache[key]
    names_hash = hashlib.md5(
        ",".join(sorted(r["name"] for r in rounds)).encode()).hexdigest()[:12]
    disk_file = HEATMAP_DISK_CACHE / f"{names_hash}_s{seat_filter or 'all'}.png"
    if disk_file.exists():
        data = disk_file.read_bytes()
        _heatmap_cache[key] = data
        return data

    bg_path = _first_frame(rounds[0]["name"])
    bg = cv2.imread(bg_path) if bg_path else None
    if bg is None:
        bg = np.zeros((720, 1280, 3), dtype=np.uint8)

    d1 = np.zeros(bg.shape[:2], dtype=np.float32)
    d2 = np.zeros(bg.shape[:2], dtype=np.float32)
    for r in rounds:
        data = _round_data(r["name"])
        if data is None:
            continue
        for sid in range(1, 8):
            if seat_filter is not None and sid != seat_filter:
                continue
            corners = data["seat_obbs"].get(str(sid)) or data["seat_obbs"].get(sid)
            for idx, c in enumerate(_raw_initial_centers_fast(data, sid)):
                if corners and not point_in_obb(c, corners):
                    continue
                target = d1 if idx == 0 else d2
                x, y = int(c[0]), int(c[1])
                if 0 <= x < bg.shape[1] and 0 <= y < bg.shape[0]:
                    target[y, x] += 1.0

    overlay = np.zeros_like(bg, dtype=np.float32)
    for density, color in [(d1, CARD1_COLOR_BGR), (d2, CARD2_COLOR_BGR)]:
        if density.max() <= 0:
            continue
        smooth = cv2.GaussianBlur(density, (51, 51), 12)
        smooth = smooth / smooth.max()
        col = np.array(color, dtype=np.float32)
        for i in range(3):
            overlay[:, :, i] += smooth * col[i] * 1.4

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    out = cv2.addWeighted(bg, 0.3, overlay, 1.0, 0)
    ok, buf = cv2.imencode(".png", out)
    img_bytes = buf.tobytes() if ok else None
    _heatmap_cache[key] = img_bytes
    if img_bytes:
        disk_file.write_bytes(img_bytes)
    return img_bytes


DEAL_ORDER = [(sid, idx) for idx in (0, 1) for sid in range(1, 8)]
INITIAL_HAND_CACHE = Path("/tmp/initial_hand_cache")
INITIAL_HAND_CACHE.mkdir(exist_ok=True)


def _initial_hand_image(round_name, step):
    data = _round_data(round_name)
    if data is None:
        return None
    order = _deal_order_for(data)
    step = max(0, min(len(order), step))
    cache_file = INITIAL_HAND_CACHE / f"{round_name}_step{step}.png"
    if cache_file.exists():
        return cache_file.read_bytes()
    bg_path = _first_frame(round_name)
    img = cv2.imread(bg_path) if bg_path else None
    if img is None:
        return None
    img = img.copy()
    for sid, oid in order[:step]:
        centers = _initial_centers(data, sid)
        if oid >= len(centers):
            continue
        center = centers[oid]
        poly = _build_polygon_at(sid, center)
        pts = np.array([[int(p[0]), int(p[1])] for p in poly],
                       dtype=np.int32).reshape(-1, 1, 2)
        col = CARD1_COLOR_BGR if oid == 0 else CARD2_COLOR_BGR
        cv2.polylines(img, [pts], True, (0, 0, 0), 5)
        cv2.polylines(img, [pts], True, col, 3)
        cx, cy = int(center[0]), int(center[1])
        cv2.circle(img, (cx, cy), 4, col, -1)
        prefix = "D" if sid == DEALER_SEAT_ID else f"P{sid}"
        tag = f"{prefix}#{oid + 1}"
        cv2.putText(img, tag, (cx + 8, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
        cv2.putText(img, tag, (cx + 8, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    ok, buf = cv2.imencode(".png", img)
    img_bytes = buf.tobytes() if ok else None
    if img_bytes:
        cache_file.write_bytes(img_bytes)
    return img_bytes


@app.route("/initial_hand/<name>/<int:step>.png")
def initial_hand_png(name, step):
    img_bytes = _initial_hand_image(name, step)
    if img_bytes is None:
        return "not found", 404
    return send_file(io.BytesIO(img_bytes), mimetype="image/png")


@app.route("/heatmap.png")
def heatmap_png():
    seat = request.args.get("seat")
    seat_filter = int(seat) if seat and seat.isdigit() else None
    img_bytes = _heatmap_image(seat_filter)
    if img_bytes is None:
        return "no rounds", 404
    return send_file(io.BytesIO(img_bytes), mimetype="image/png")


_annot_upload_dir = Path(POLYDRAW_PATH).parent / "uploads"
_annot_upload_dir.mkdir(parents=True, exist_ok=True)


@app.route("/annotate")
def annotate_page():
    existing = {}
    try:
        with open(POLYDRAW_PATH) as f:
            d = json.load(f)
        by_id = {p["id"]: [p["x"], p["y"]] for p in d["points"]}
        for g in d["pointGroups"]:
            existing[g["name"]] = [by_id[i] for i in g["pointIds"]]
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        pass
    return render_template_string(ANNOTATE_HTML, existing_json=json.dumps(existing))


@app.route("/annotate/upload", methods=["POST"])
def annotate_upload():
    f = request.files.get("image")
    if f is None or not f.filename:
        return jsonify({"error": "no file"}), 400
    safe = Path(f.filename).name
    path = _annot_upload_dir / safe
    f.save(str(path))
    return jsonify({"url": f"/annotate/img/{safe}"})


@app.route("/annotate/img/<name>")
def annotate_img(name):
    safe = Path(name).name
    return send_file(str(_annot_upload_dir / safe))


@app.route("/annotate/save", methods=["POST"])
def annotate_save():
    global _polydraw_cache, _initial_cache
    body = request.get_json() or {}
    polys = body.get("polygons", {})
    points, groups = [], []
    n = 0
    for seat_name, corners in polys.items():
        if not (isinstance(corners, list) and len(corners) == 4):
            continue
        pids = []
        for c in corners:
            pid = f"a{n}"; n += 1
            points.append({"id": pid, "x": float(c[0]), "y": float(c[1])})
            pids.append(pid)
        groups.append({"id": f"g{seat_name}", "pointIds": pids, "name": seat_name})
    with open(POLYDRAW_PATH, "w") as f:
        json.dump({"points": points, "pointGroups": groups, "superGroups": []}, f)
    _polydraw_cache = None
    _initial_cache = {}
    return jsonify({"ok": True, "saved_to": POLYDRAW_PATH, "n_seats": len(groups)})


INDEX_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Card placement predictor</title>
<style>
body { font-family: -apple-system, sans-serif; margin: 16px; background:#1c1c1c; color:#eee; }
h1 { margin: 0 0 12px; font-size: 18px; }
.row { display: flex; gap: 10px; align-items: center; margin-bottom: 10px; flex-wrap: wrap; }
select, button { font-size: 13px; padding: 5px 10px; border-radius: 4px; border: 1px solid #555; background:#2a2a2a; color:#eee; }
button { cursor: pointer; }
button:disabled { opacity: 0.35; cursor: not-allowed; }
.layout { display: flex; gap: 16px; align-items: flex-start; }
.pane { background:#262626; padding: 10px; border-radius: 6px; }
#vis { display:block; max-width: 100%; }
table { border-collapse: collapse; }
th, td { padding: 6px 10px; border: 1px solid #404040; }
th { background:#303030; text-align:left; }
.hit { background:#fff; color:#000; border-color:#fff; }
.double { background:#f5a200; color:#000; border-color:#f5a200; }
.reset { background:#c0392b; color:#fff; border-color:#c0392b; }
.save { background:#2e8b57; color:#fff; border-color:#2e8b57; }
#log { font-family: monospace; font-size: 12px; max-height: 240px; overflow-y: auto; background:#1a1a1a; padding: 8px; border-radius: 4px; margin-top: 10px; }
.muted { color: #888; }
</style></head><body>
<h1>Card placement predictor</h1>
<div class="row">
  <label><input type="checkbox" id="heatmap-toggle"> Initial-card heatmap overlay</label>
  <span style="display:inline-block;width:12px;height:12px;background:#dc3c3c;border-radius:2px;margin-left:8px;"></span>
  <span class="muted">Card 1</span>
  <span style="display:inline-block;width:12px;height:12px;background:#3cc8dc;border-radius:2px;margin-left:8px;"></span>
  <span class="muted">Card 2</span>
</div>
<div class="row">
  <label>Round:
    <select id="round-select">
      <option value="">— pick a round —</option>
      {% for r in rounds %}
      <option value="{{r.name}}">{{r.name}} ({{r.counts}})</option>
      {% endfor %}
    </select>
  </label>
  <button id="initial-hand-btn" disabled>Initial hand ▶</button>
  <button id="save-btn" class="save" disabled>Save</button>
  <button id="reset-btn" class="reset" disabled>Reset session</button>
  <span class="muted">{{rounds|length}} rounds available</span>
</div>
<div class="layout">
  <div class="pane"><img id="vis" src=""></div>
  <div class="pane" style="min-width:340px">
    <table id="seat-table">
      <thead><tr><th>Seat</th><th># cards</th><th>Hit</th><th>Double</th></tr></thead>
      <tbody></tbody>
    </table>
    <div id="log"></div>
  </div>
</div>
<script>
const sel = document.getElementById("round-select");
const tbody = document.querySelector("#seat-table tbody");
const visImg = document.getElementById("vis");
const resetBtn = document.getElementById("reset-btn");
const initBtn = document.getElementById("initial-hand-btn");
const saveBtn = document.getElementById("save-btn");
const logDiv = document.getElementById("log");
let currentRound = "";

function log(msg) {
  const t = new Date().toLocaleTimeString();
  logDiv.innerHTML = `[${t}] ${msg}<br>` + logDiv.innerHTML;
}

function anyInitialPlaced(r) {
  if (!r || !r.seats) return false;
  return Object.values(r.seats).some(s => s.n_cards >= 2);
}
function applyState(r) {
  renderSeats(r.seats);
  saveBtn.disabled = !anyInitialPlaced(r);
  refreshImage();
}

async function loadRound(name) {
  currentRound = name;
  if (!name) { tbody.innerHTML=""; visImg.src=""; resetBtn.disabled=true; initBtn.disabled=true; saveBtn.disabled=true; logDiv.innerHTML=""; return; }
  const r = await (await fetch(`/round/${name}`)).json();
  applyState(r);
  resetBtn.disabled = false;
  initBtn.disabled = false;
  saveBtn.disabled = !anyInitialPlaced(r);
  log(`Loaded round ${name}`);
}
initBtn.addEventListener("click", async () => {
  if (!currentRound) return;
  initBtn.disabled = true;
  document.getElementById("heatmap-toggle").checked = false;
  const meta = await (await fetch(`/round/${currentRound}`)).json();
  const total = meta.initial_hand_steps || 0;
  if (!total) { log("no players in this round"); initBtn.disabled = false; return; }
  for (let step = 1; step <= total; step++) {
    visImg.src = `/initial_hand/${currentRound}/${step}.png?t=${Date.now()}`;
    await new Promise(r => setTimeout(r, 350));
  }
  const r = await (await fetch(`/place_initial/${currentRound}`, {method:"POST"})).json();
  applyState(r);
  log(`Initial hand placed for P${meta.present_seats.join(", P")}`);
  initBtn.disabled = false;
});
function refreshImage() {
  const heat = document.getElementById('heatmap-toggle').checked;
  if (heat) {
    visImg.src = `/heatmap.png?t=${Date.now()}`;
  } else if (currentRound) {
    visImg.src = `/vis/${currentRound}.png?t=${Date.now()}`;
  } else {
    visImg.src = "";
  }
}
document.getElementById('heatmap-toggle').addEventListener('change', refreshImage);
function renderSeats(seats) {
  tbody.innerHTML = "";
  for (const sid of [1,2,3,4,5,6,7,0]) {
    const s = seats[sid] || {n_cards:0, can_double:false, can_hit:false};
    const label = sid === 0 ? "Dealer" : `P${sid}`;
    const doubleCell = sid === 0
      ? `<td class="muted">—</td>`
      : `<td><button class="double" data-sid="${sid}" data-action="double" ${s.can_double?"":"disabled"}>Double</button></td>`;
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${label}</td>
      <td>${s.n_cards}</td>
      <td><button class="hit" data-sid="${sid}" data-action="hit" ${s.can_hit?"":"disabled"}>Hit</button></td>
      ${doubleCell}`;
    tbody.appendChild(tr);
  }
  tbody.querySelectorAll("button").forEach(b => b.addEventListener("click", onAct));
}
async function onAct(e) {
  const sid = e.target.dataset.sid;
  const action = e.target.dataset.action;
  const resp = await fetch(`/predict/${currentRound}`, {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({seat: sid, action}),
  });
  if (!resp.ok) { const er = await resp.json(); log(`<span style="color:#f88">err: ${er.error}</span>`); return; }
  const r = await resp.json();
  const la = r.last_action;
  const seatLbl = la.seat == 0 ? 'Dealer' : `P${la.seat}`;
  log(`${seatLbl} ${la.is_double?'Double':'Hit'} → center=(${la.center[0].toFixed(0)}, ${la.center[1].toFixed(0)}) angle=${la.chosen_angle_deg.toFixed(0)}°`);
  applyState(r);
}
sel.addEventListener("change", e => loadRound(e.target.value));
saveBtn.addEventListener("click", async () => {
  if (!currentRound) return;
  saveBtn.disabled = true;
  const r = await (await fetch(`/save/${currentRound}`, {method:"POST"})).json();
  if (r.ok) log(`Saved ${r.seats_saved} seats → ${r.json}`);
  else log(`<span style="color:#f88">save err: ${r.error}</span>`);
  saveBtn.disabled = false;
});
resetBtn.addEventListener("click", async () => {
  if (!currentRound) return;
  const r = await (await fetch(`/reset/${currentRound}`, {method:"POST"})).json();
  applyState(r);
  log(`Reset ${currentRound}`);
});
</script></body></html>"""


ANNOTATE_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Annotate seat polygons</title>
<style>
body { font-family: -apple-system, sans-serif; margin: 16px; background:#1c1c1c; color:#eee; }
h1 { margin: 0 0 12px; font-size: 18px; }
.row { display: flex; gap: 10px; align-items: center; margin: 8px 0; flex-wrap: wrap; }
button, select, input[type=file] { font-size: 13px; padding: 5px 10px; border-radius: 4px; border: 1px solid #555; background:#2a2a2a; color:#eee; }
button { cursor: pointer; }
button:disabled { opacity: 0.4; cursor: not-allowed; }
.save { background:#2e8b57; color:#fff; border-color:#2e8b57; }
.clear { background:#c0392b; color:#fff; border-color:#c0392b; }
.layout { display: flex; gap: 16px; align-items: flex-start; }
.pane { background:#262626; padding: 10px; border-radius: 6px; }
canvas { display:block; max-width: 100%; cursor: crosshair; image-rendering: -webkit-optimize-contrast; }
table { border-collapse: collapse; }
th, td { padding: 4px 8px; border: 1px solid #404040; font-size: 12px; }
th { background:#303030; text-align:left; }
.done { color:#7fd17f; }
.partial { color:#f5b840; }
.empty { color:#888; }
.muted { color:#888; font-size: 12px; }
#status { font-family: monospace; font-size: 12px; min-height: 16px; }
</style></head><body>
<h1>Annotate seat card polygons</h1>
<div class="row">
  <label>Image: <input type="file" id="img-file" accept="image/*"></label>
  <label>Seat:
    <select id="seat-select">
      <option value="p1">P1</option><option value="p2">P2</option>
      <option value="p3">P3</option><option value="p4">P4</option>
      <option value="p5">P5</option><option value="p6">P6</option>
      <option value="p7">P7</option>
      <option value="dealer">Dealer</option>
    </select>
  </label>
  <button id="clear-seat" class="clear">Clear current seat</button>
  <button id="clear-all" class="clear">Clear all</button>
  <button id="save" class="save">Save</button>
  <span class="muted">Click 4 corners per seat. Next click after 4th replaces oldest.</span>
</div>
<div id="status"></div>
<div class="layout">
  <div class="pane"><canvas id="canvas"></canvas></div>
  <div class="pane" style="min-width:200px">
    <table><thead><tr><th>Seat</th><th># corners</th></tr></thead>
      <tbody id="status-table"></tbody></table>
  </div>
</div>
<script>
const fileInput = document.getElementById('img-file');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const seatSel = document.getElementById('seat-select');
const status = document.getElementById('status');
const statusTable = document.getElementById('status-table');
const SEATS = ['p1','p2','p3','p4','p5','p6','p7','dealer'];
const COL = {p1:'#dc3c3c', p2:'#f09650', p3:'#f0dc3c', p4:'#50c850', p5:'#3cc8dc', p6:'#3c64dc', p7:'#c83cc8', dealer:'#ffffff'};
let img = null;
let polygons = {{ existing_json|safe }};
for (const s of SEATS) if (!polygons[s]) polygons[s] = [];

fileInput.addEventListener('change', async (e) => {
  const f = e.target.files[0]; if (!f) return;
  const fd = new FormData(); fd.append('image', f);
  const r = await fetch('/annotate/upload', {method:'POST', body: fd}).then(r=>r.json());
  if (r.error) { alert(r.error); return; }
  img = new Image();
  img.onload = () => { canvas.width = img.naturalWidth; canvas.height = img.naturalHeight; redraw(); };
  img.src = r.url;
});

canvas.addEventListener('click', (e) => {
  if (!img) return;
  const rect = canvas.getBoundingClientRect();
  const sx = canvas.width / rect.width, sy = canvas.height / rect.height;
  const x = (e.clientX - rect.left) * sx;
  const y = (e.clientY - rect.top) * sy;
  const seat = seatSel.value;
  const pts = polygons[seat];
  if (pts.length >= 4) pts.shift();
  pts.push([x, y]);
  redraw();
});

document.getElementById('clear-seat').addEventListener('click', () => {
  polygons[seatSel.value] = []; redraw();
});
document.getElementById('clear-all').addEventListener('click', () => {
  for (const s of SEATS) polygons[s] = []; redraw();
});
document.getElementById('save').addEventListener('click', async () => {
  const r = await fetch('/annotate/save', {method:'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({polygons})}).then(r=>r.json());
  status.textContent = r.ok ? `Saved ${r.n_seats} seat polygons to ${r.saved_to}` : `Error: ${r.error}`;
});

function ccwSort(pts) {
  const cx = pts.reduce((a,p)=>a+p[0],0)/pts.length;
  const cy = pts.reduce((a,p)=>a+p[1],0)/pts.length;
  return [...pts].sort((a,b)=>Math.atan2(a[1]-cy,a[0]-cx)-Math.atan2(b[1]-cy,b[0]-cx));
}

function redraw() {
  if (!img) return;
  ctx.drawImage(img, 0, 0);
  for (const s of SEATS) {
    const pts = polygons[s]; if (!pts.length) continue;
    const col = COL[s];
    if (pts.length === 4) {
      const ord = ccwSort(pts);
      ctx.strokeStyle = '#000'; ctx.lineWidth = 5;
      ctx.beginPath(); ord.forEach((p,i) => i===0 ? ctx.moveTo(p[0],p[1]) : ctx.lineTo(p[0],p[1]));
      ctx.closePath(); ctx.stroke();
      ctx.strokeStyle = col; ctx.lineWidth = 3;
      ctx.beginPath(); ord.forEach((p,i) => i===0 ? ctx.moveTo(p[0],p[1]) : ctx.lineTo(p[0],p[1]));
      ctx.closePath(); ctx.stroke();
    }
    for (const p of pts) {
      ctx.fillStyle = col;
      ctx.beginPath(); ctx.arc(p[0], p[1], 5, 0, 2*Math.PI); ctx.fill();
      ctx.strokeStyle = '#000'; ctx.lineWidth = 1; ctx.stroke();
    }
    ctx.fillStyle = col; ctx.strokeStyle = '#000'; ctx.lineWidth = 3;
    const lp = pts[pts.length-1];
    ctx.font = '14px sans-serif';
    ctx.strokeText(s.toUpperCase(), lp[0]+8, lp[1]-8);
    ctx.fillText(s.toUpperCase(), lp[0]+8, lp[1]-8);
  }
  renderTable();
}
function renderTable() {
  statusTable.innerHTML = '';
  for (const s of SEATS) {
    const n = polygons[s].length;
    const cls = n===4 ? 'done' : (n>0 ? 'partial' : 'empty');
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${s.toUpperCase()}</td><td class="${cls}">${n}/4</td>`;
    statusTable.appendChild(tr);
  }
}
renderTable();
</script></body></html>"""


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)
