"""Intra-pile post-processing: take per-frame card detections + per-round seat
OBBs and emit per-card labels <player_idx, ordinal, double>.

Heuristics (v1):
- Tracking: greedy nearest-neighbor by rank+suit + spatial proximity across frames.
- Seat: point-in-OBB on the card's polygon_center using the YOLO seat polygons.
- Ordinal: 1-indexed order of first-appearance frames within each seat.
- Double: a card whose AABB aspect ratio is "wide" (w/h > 1.1) AND there are >= 3
  cards in the seat. Marks the 3rd+ card that's clearly perpendicular.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------- card detection helpers ----------

def _center(det: dict) -> Tuple[float, float]:
    pc = det["polygon_center"]
    if isinstance(pc[0], list):
        pc = pc[0]
    return float(pc[0]), float(pc[1])


def _aabb_aspect(det: dict) -> float:
    """width / height of the axis-aligned box (>1 means landscape/perpendicular)."""
    x1, y1, x2, y2 = det["box"]
    w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    return w / h


# ---------- tracking ----------

def track_cards(
    frames: Sequence[dict],
    spatial_thresh_px: float = 35.0,
    recency_frames: int = 75,
    min_observations: int = 30,
    min_duration_frames: int = 50,
) -> List[dict]:
    """Stitch per-frame detections into per-card tracks using **spatial only**
    association (rank/suit OCR is too noisy to gate matching on).

    Args:
      frames: list of {"frame_id": int, "detections": [det,...]}.
      spatial_thresh_px: max center movement between consecutive observations.
      recency_frames: drop matches if the track hasn't been seen for this long
        (cards are static once placed, so reappearing far away ⇒ different card).
      min_observations: discard tracks with fewer observations than this
        (transient spurious detections).

    Returns:
      list of tracks. Each track:
        {
          "track_id": int,
          "rank": str, "suit": str,    # majority vote across observations
          "first_frame": int, "last_frame": int,
          "centers": [(x,y), ...],
          "boxes":   [[x1,y1,x2,y2], ...],
          "confs":   [float, ...],
          "rsuit":   [(rank, suit), ...],
          "first_center": (x,y),
        }
    """
    from collections import Counter

    tracks: List[dict] = []
    for fr in frames:
        fid = fr["frame_id"]
        dets = fr.get("detections") or []
        # Greedy assignment: sort detections by confidence so the best gets first pick
        dets_sorted = sorted(dets, key=lambda d: -float(d.get("conf", 0.0)))
        claimed_tracks = set()
        for d in dets_sorted:
            cx, cy = _center(d)
            rs = (d.get("rank", "?"), d.get("suit", "?"))
            best, best_dist = None, math.inf
            for t in tracks:
                if t["track_id"] in claimed_tracks:
                    continue
                if fid - t["last_frame"] > recency_frames:
                    continue
                lx, ly = t["centers"][-1]
                dd = math.hypot(cx - lx, cy - ly)
                if dd < best_dist:
                    best_dist = dd
                    best = t
            if best is not None and best_dist <= spatial_thresh_px:
                best["last_frame"] = fid
                best["obs_frames"].append(fid)
                best["centers"].append((cx, cy))
                best["boxes"].append(d["box"])
                best["confs"].append(float(d.get("conf", 0.0)))
                best["rsuit"].append(rs)
                claimed_tracks.add(best["track_id"])
            else:
                tracks.append({
                    "track_id": len(tracks),
                    "rank": rs[0], "suit": rs[1],
                    "first_frame": fid, "last_frame": fid,
                    "obs_frames": [fid],
                    "centers": [(cx, cy)],
                    "boxes": [d["box"]],
                    "confs": [float(d.get("conf", 0.0))],
                    "rsuit": [rs],
                    "first_center": (cx, cy),
                })

    # Filter noisy tracks: must persist long enough and have enough observations
    # (displacement filter dropped; the duration + obs filters catch the obvious
    # transient artefacts and we don't want to reject real cards that legitimately
    # moved, e.g. the dealer flipping the hole card).
    def _ok(t):
        if len(t["centers"]) < min_observations:
            return False
        if t["last_frame"] - t["first_frame"] < min_duration_frames:
            return False
        return True
    tracks = [t for t in tracks if _ok(t)]
    # Re-id and majority-vote rank/suit
    for i, t in enumerate(tracks):
        t["track_id"] = i
        c = Counter(t["rsuit"])
        (r_top, s_top), _ = c.most_common(1)[0]
        t["rank"], t["suit"] = r_top, s_top
    return tracks


# ---------- seat assignment via OBB ----------

def point_in_obb(p: Tuple[float, float], corners: Sequence[Sequence[float]]) -> bool:
    """Rotate the point into the box's local frame and test as axis-aligned.

    corners: 4 [x,y] points in CW or CCW order. The first edge (c0->c1) defines
    the box's local x-axis; the perpendicular edge (c0->c3) defines local y.

    Cost: ~4 mults + 4 adds + 2 comparisons. Functionally identical to the
    cross-product side test, but matches the rotate-to-local explanation.
    """
    px, py = p
    c0x, c0y = corners[0]
    c1x, c1y = corners[1]
    c3x, c3y = corners[3]
    # Local axes (not unit-normalised yet)
    ex_x, ex_y = c1x - c0x, c1y - c0y          # along width
    ey_x, ey_y = c3x - c0x, c3y - c0y          # along height
    # Vector from origin (c0) to point
    dx, dy = px - c0x, py - c0y
    # Project onto each axis (gives length along that axis, scaled by axis length)
    proj_x = dx * ex_x + dy * ex_y
    proj_y = dx * ey_x + dy * ey_y
    # Axis squared-lengths (avoids a sqrt; same as |edge|^2)
    len_x_sq = ex_x * ex_x + ex_y * ex_y
    len_y_sq = ey_x * ey_x + ey_y * ey_y
    return 0.0 <= proj_x <= len_x_sq and 0.0 <= proj_y <= len_y_sq


def _point_to_segment_dist(p, a, b):
    px, py = p
    ax, ay = a
    bx, by = b
    abx, aby = bx - ax, by - ay
    denom = abx * abx + aby * aby
    if denom == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / denom))
    qx, qy = ax + t * abx, ay + t * aby
    return math.hypot(px - qx, py - qy)


def point_to_obb_distance(p, corners):
    """0 if inside, else min distance to any of the 4 edges (in pixels)."""
    if point_in_obb(p, corners):
        return 0.0
    return min(
        _point_to_segment_dist(p, corners[i], corners[(i + 1) % 4])
        for i in range(4)
    )


def _assign_one_point(p, seat_obbs, fallback_dist_px):
    """Return (seat_id or None, 'inside'/'nearest(N)'/None)."""
    for sid, corners in seat_obbs.items():
        if point_in_obb(p, corners):
            return sid, "inside"
    best_sid, best_d = None, math.inf
    for sid, corners in seat_obbs.items():
        d = point_to_obb_distance(p, corners)
        if d < best_d:
            best_d = d
            best_sid = sid
    if best_sid is not None and best_d <= fallback_dist_px:
        return best_sid, f"nearest({best_d:.1f}px)"
    return None, None


def assign_seats(
    tracks: List[dict],
    seat_obbs: Dict[int, List[List[float]]],
    src_w: int, src_h: int,
    img_w: int, img_h: int,
    fallback_dist_px: float = 30.0,
) -> None:
    """In-place: set track['seat'] (int 0..7) or None via per-observation
    majority vote.

    For each frame the card was observed:
      - Point-in-OBB test against every seat. If hit, that observation votes.
      - Else nearest-OBB fallback within fallback_dist_px casts a vote.
      - Otherwise the observation votes None.

    The seat with the most votes wins. This is robust to the card's detected
    centroid drifting across the boundary between two adjacent seats over time.
    """
    from collections import Counter

    sx = img_w / src_w
    sy = img_h / src_h
    for t in tracks:
        votes = Counter()
        for cx, cy in t["centers"]:
            p = (cx * sx, cy * sy)
            sid, _ = _assign_one_point(p, seat_obbs, fallback_dist_px)
            votes[sid] += 1
        # Pick the seat with the most votes (None counted, but only wins if it
        # truly dominates)
        seat, n = votes.most_common(1)[0]
        t["seat"] = seat
        if seat is None:
            t["seat_via"] = None
        else:
            inside_votes = sum(
                1 for c in t["centers"]
                if point_in_obb((c[0] * sx, c[1] * sy), seat_obbs[seat])
            )
            t["seat_via"] = "inside" if inside_votes > n / 2 else "nearest"
        t["seat_vote_share"] = n / len(t["centers"])


# ---------- ordinal ----------

def assign_ordinals(tracks: List[dict],
                    seat_obbs: Dict[int, List[List[float]]] = None) -> None:
    """Within each seat, assign ordinal=1..N by the order in which each card
    first appears INSIDE its assigned seat's OBB.

    Using the seat-region first-frame (not the track's overall first_frame)
    handles cases where the same track had earlier detections elsewhere on
    the table that aren't part of this player's dealing sequence.
    """
    by_seat: Dict[int, List[dict]] = {}
    for t in tracks:
        s = t.get("seat")
        if s is None:
            t["ordinal"] = None
            continue
        by_seat.setdefault(s, []).append(t)

    def first_frame_in_seat(t, corners):
        if corners is None or "obs_frames" not in t:
            return t["first_frame"]
        for fid, (cx, cy) in zip(t["obs_frames"], t["centers"]):
            if point_in_obb((cx, cy), corners):
                return fid
        return t["first_frame"]

    for s, ts in by_seat.items():
        corners = seat_obbs.get(s) if seat_obbs else None
        ts.sort(key=lambda t: first_frame_in_seat(t, corners))
        for i, t in enumerate(ts, 1):
            t["ordinal"] = i


# ---------- double ----------

def _project_to_obb_long_axis(p, corners):
    """Project p onto the OBB's *long* axis (length along the fan direction)."""
    return _project_to_obb_axis(p, corners, want_long=True)


def _project_to_obb_short_axis(p, corners):
    """Project p onto the OBB's *short* axis (length perpendicular to the fan).
    This is the X axis of the rotated-to-vertical crop. In a normal fan,
    cards' short-axis projections are clustered tightly; a perpendicular
    double card sits visibly offset along this axis."""
    return _project_to_obb_axis(p, corners, want_long=False)


def _project_to_obb_axis(p, corners, want_long: bool):
    c0, c1, c3 = corners[0], corners[1], corners[3]
    e01 = (c1[0] - c0[0], c1[1] - c0[1])
    e03 = (c3[0] - c0[0], c3[1] - c0[1])
    L01 = math.hypot(*e01)
    L03 = math.hypot(*e03)
    long_is_01 = L01 >= L03
    if want_long:
        ex, len_ex = (e01, L01) if long_is_01 else (e03, L03)
    else:
        ex, len_ex = (e03, L03) if long_is_01 else (e01, L01)
    if len_ex == 0:
        return 0.0
    d = (p[0] - c0[0], p[1] - c0[1])
    return (d[0] * ex[0] + d[1] * ex[1]) / len_ex


def assign_doubles_from_frame(
    tracks: List[dict],
    seat_obbs: Dict[int, List[List[float]]],
    frame_detections: list,
    sin_thresh: float = 0.3,
    min_step_px: float = 4.0,
) -> None:
    """Simple double detection using ONE frame's detections + centroids.

    Rule (absolute-side test): draw the fan vector v_1 from card 1's center to
    card 2's center. For each subsequent card i (i >= 3), compute |sin(angle)|
    between v_1 and v_i (= |cross|/(|v_1|*|v_i|)).

      |sin_angle| > sin_thresh   →  card_i deviates meaningfully from the
                                    fan (either side) → flag as double.
      step length < min_step_px →  stacked on top → flag as double.

    Using |sin_a| avoids the image-coord sign convention issue: the
    perpendicular double card can land on either side of the fan depending
    on which seat / how the dealer reached in.

    Steps:
      1. For each player seat OBB, collect detections in this frame whose
         centroid falls inside the OBB (filters duplicates/transit cards).
      2. Sort those detections by appearance order in the seat's tracks
         (using each card's matching track ordinal).
      3. Compute consecutive direction vectors and check angle changes.

    Skips Dealer.
    """
    for t in tracks:
        t["double"] = False

    # Bucket detections by seat (centroid inside OBB)
    by_seat: Dict[int, list] = {}
    for det in frame_detections:
        pc = det.get("polygon_center")
        if pc is None:
            continue
        if isinstance(pc[0], list):
            pc = pc[0]
        rank, suit = det.get("rank", "?"), det.get("suit", "?")
        if (rank, suit) == ("CB", "CB"):
            continue
        for sid, corners in seat_obbs.items():
            if sid == 0:
                continue
            if point_in_obb(pc, corners):
                by_seat.setdefault(sid, []).append((pc, rank, suit))
                break

    for sid, dets in by_seat.items():
        if len(dets) < 3:
            continue
        seat_tracks = [t for t in tracks if t.get("seat") == sid]
        # Match each detection to a seat track using (rank, suit) AND
        # centroid proximity, so duplicate-identity cards (same rank+suit at
        # different table positions) get their distinct ordinals.
        def det_ordinal(det):
            pc, r, su = det
            best_ord, best_d = 99, math.inf
            for t in seat_tracks:
                if (t["rank"], t["suit"]) != (r, su):
                    continue
                tc = t["centers"][-1]
                d = math.hypot(pc[0] - tc[0], pc[1] - tc[1])
                if d < best_d:
                    best_d = d
                    best_ord = t.get("ordinal") or 99
            return best_ord
        dets_ord = sorted(dets, key=det_ordinal)
        centers = [d[0] for d in dets_ord]
        # Fan direction from first two cards
        v0 = (centers[1][0] - centers[0][0], centers[1][1] - centers[0][1])
        len_v0 = math.hypot(*v0)
        if len_v0 < min_step_px:
            continue
        for i in range(2, len(centers)):
            vi = (centers[i][0] - centers[i - 1][0],
                  centers[i][1] - centers[i - 1][1])
            len_vi = math.hypot(*vi)
            if len_vi < min_step_px:
                cand = max(seat_tracks, key=lambda t: t.get("ordinal") or 0)
                cand["double"] = True
                break
            # Signed sine of the angle between v0 and vi (image coords, y down)
            sin_a = (v0[0] * vi[1] - v0[1] * vi[0]) / (len_v0 * len_vi)
            if sin_a < -sin_thresh:
                cand = max(seat_tracks, key=lambda t: t.get("ordinal") or 0)
                cand["double"] = True
                break


def assign_doubles(tracks: List[dict],
                   seat_obbs: Dict[int, List[List[float]]] = None,
                   min_step_px: float = 3.0,
                   current_fid: int = None) -> None:
    """Mark a card as double=True if the cards' projections on the seat OBB's
    *short axis* (the axis perpendicular to the fan direction) zigzag, i.e.
    go right-left-right (or left-right-left).

    Intuition: in a normal fan, every card sits roughly centered along the
    short axis of the seat region — short-axis projections are clustered.
    A double card is placed perpendicular ON TOP of the fan, so its
    short-axis projection is visibly offset relative to the others. When
    you watch the short-axis positions in ordinal order, a real double
    creates an alternation pattern (the perpendicular card breaks the line).

    The first card whose short-axis step has the opposite sign to the
    previous step is flagged as the double.

    Skips: Dealer, seats with < 3 cards.
    """
    by_seat: Dict[int, List[dict]] = {}
    for t in tracks:
        t["double"] = False
        s = t.get("seat")
        if s is None:
            continue
        by_seat.setdefault(s, []).append(t)

    if seat_obbs is None:
        return

    for s, ts in by_seat.items():
        if s == 0:
            continue
        corners = seat_obbs.get(s) or seat_obbs.get(str(s))
        if corners is None:
            continue
        def center_at_fid(t, fid):
            if fid is None or "obs_frames" not in t:
                return t["centers"][-1]
            of, cs = t["obs_frames"], t["centers"]
            i = min(range(len(of)), key=lambda k: abs(of[k] - fid))
            return cs[i]
        # Keep only cards that (a) are alive at the current frame and
        # (b) actually sit inside this seat's OBB at that frame.
        # (b) filters out duplicate-detection ghosts that got assigned via
        # majority vote but whose rep-frame centroid is outside the OBB.
        if current_fid is not None:
            kept = []
            for t in ts:
                if not (t["first_frame"] <= current_fid <= t["last_frame"]):
                    continue
                c = center_at_fid(t, current_fid)
                if point_in_obb(c, corners):
                    kept.append(t)
            ts = kept
        if len(ts) < 3:
            continue
        ts_sorted = sorted(ts, key=lambda t: (t.get("ordinal") or t["first_frame"]))
        long_xs = [_project_to_obb_long_axis(center_at_fid(t, current_fid), corners)
                   for t in ts_sorted]
        short_xs = [_project_to_obb_short_axis(center_at_fid(t, current_fid), corners)
                    for t in ts_sorted]

        def first_sign_reversal(xs):
            prev_sign = 0
            for i in range(1, len(xs)):
                step = xs[i] - xs[i - 1]
                if abs(step) < min_step_px:
                    continue
                sign = 1 if step > 0 else -1
                if prev_sign != 0 and sign != prev_sign:
                    return i
                prev_sign = sign
            return None

        rev_long = first_sign_reversal(long_xs)
        rev_short = first_sign_reversal(short_xs)
        candidates = [r for r in (rev_long, rev_short) if r is not None]
        if candidates:
            # Earliest reversal wins (the card that broke the line)
            idx = min(candidates)
            ts_sorted[idx]["double"] = True


# ---------- top-level convenience ----------

def dedup_frame_detections(dets, dist_px: float = 20.0):
    """Remove near-duplicate detections from a single frame.

    Two detections are considered duplicates if they share (rank, suit) AND
    their polygon_centers are within `dist_px` of each other. Keep the
    highest-confidence one and drop the rest.
    """
    if not dets:
        return dets
    sorted_dets = sorted(dets, key=lambda d: -float(d.get("conf", 0.0)))
    kept = []
    for d in sorted_dets:
        pc = d.get("polygon_center")
        if pc is None:
            continue
        if isinstance(pc[0], list):
            pc = pc[0]
        rs = (d.get("rank", "?"), d.get("suit", "?"))
        dup = False
        for k in kept:
            kpc = k.get("polygon_center")
            if isinstance(kpc[0], list):
                kpc = kpc[0]
            krs = (k.get("rank", "?"), k.get("suit", "?"))
            if rs == krs and math.hypot(pc[0] - kpc[0], pc[1] - kpc[1]) < dist_px:
                dup = True
                break
        if not dup:
            kept.append(d)
    return kept


def label_round(
    frames: Sequence[dict],
    seat_obbs: Dict[int, List[List[float]]],
    src_w: int = 1920, src_h: int = 1080,
    img_w: int = 1280, img_h: int = 720,
    drop_cardbacks: bool = True,
    rep_fid: int = None,
    dedup_dets: bool = True,
) -> List[dict]:
    """Run the full pipeline on one round's frames. Returns the list of tracks
    with seat/ordinal/double fields populated.

    Card-back ("CB") detections come from the dealer's shoe / deck and don't
    represent on-table playable cards, so they're dropped by default.
    """
    if dedup_dets:
        frames = [
            {**fr, "detections": dedup_frame_detections(fr.get("detections") or [])}
            for fr in frames
        ]
    tracks = track_cards(frames)
    if drop_cardbacks:
        tracks = [
            t for t in tracks
            if not (t.get("rank") == "CB" and t.get("suit") == "CB")
        ]
        for i, t in enumerate(tracks):
            t["track_id"] = i
    assign_seats(tracks, seat_obbs, src_w, src_h, img_w, img_h)
    assign_ordinals(tracks, seat_obbs=seat_obbs)
    # Simple double detection on a single representative frame's detections
    # (frames already deduped above if dedup_dets=True)
    if rep_fid is not None:
        rep_dets = next((fr.get("detections") or []
                         for fr in frames if fr.get("frame_id") == rep_fid), [])
        assign_doubles_from_frame(tracks, seat_obbs, rep_dets)
    else:
        assign_doubles(tracks, seat_obbs=seat_obbs, current_fid=rep_fid)
    return tracks
