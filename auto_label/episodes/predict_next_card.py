"""Predict where the next card will land for a given seat.

v0 := vector from initial card 1 to card 2 (always the first two by ordinal).

- Hit: next step continues v0 (angle HIT_ANGLE_DEG = 0). Every subsequent hit
  uses the same v0-based step, so cards keep extending the original fan line.
- Double: requires exactly 2 cards; next step is at DOUBLE_ANGLE_DEG from v0
  (same magnitude). Default 45.
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, "/home/ubuntu/sahithi/motion-data-process/auto_label/episodes")
from intra_pile_util import point_in_obb

DEFAULT_CARDS_DIR = ("/home/ubuntu/us-west-3-fs/live_dealer_blackjack/roundcut/"
                     "cards_results/good_quality_rounds/all_jsons")

DOUBLE_ANGLE_DEG = 60.0
DOUBLE_ANGLE_BOUNDS = (30.0, 90.0)
HIT_ANGLE_DEG = 0.0
NORMAL_MAX_TWEAK_DEG = 15.0
OVERLAP_RADIUS_PX = 25.0
ANGLE_STEP_DEG = 5.0
STEP_MAGNITUDE_PX = 30.0

SEAT_ID = {"Dealer": 0, "P1": 1, "P2": 2, "P3": 3,
           "P4": 4, "P5": 5, "P6": 6, "P7": 7}


def _seat_centers_in_order(data, seat_id):
    ts = [t for t in data["tracks"]
          if t.get("seat") == seat_id and t.get("ordinal") is not None]
    ts.sort(key=lambda t: t["ordinal"])
    return [tuple(t["last_center"]) for t in ts], ts


def _mean_step(centers):
    if len(centers) < 2:
        return None
    dxs = [centers[i][0] - centers[i - 1][0] for i in range(1, len(centers))]
    dys = [centers[i][1] - centers[i - 1][1] for i in range(1, len(centers))]
    return (sum(dxs) / len(dxs), sum(dys) / len(dys))


def _rotate(step, angle_deg):
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    return (step[0] * c + step[1] * s, -step[0] * s + step[1] * c)


def _all_centers(data):
    return [tuple(t["last_center"]) for t in data["tracks"]
            if t.get("seat") is not None]


def _rep_fid(data):
    return int(data["representative_frame"].split("=")[-1])


def _seat_boxes_at_rep(data, cards_dir, seat_id):
    rep_fid = _rep_fid(data)
    jsonl = Path(cards_dir) / f"{data['round']}_card.jsonl"
    rep_frame = None
    with open(jsonl) as f:
        for line in f:
            fr = json.loads(line)
            if fr.get("frame_id") == rep_fid:
                rep_frame = fr
                break
    if rep_frame is None:
        return []
    corners = (data["seat_obbs"].get(str(seat_id))
               or data["seat_obbs"].get(seat_id))
    if corners is None:
        return []
    boxes = []
    for det in rep_frame.get("detections") or []:
        pc = det.get("polygon_center")
        if pc is None:
            continue
        if isinstance(pc[0], list):
            pc = pc[0]
        if (det.get("rank"), det.get("suit")) == ("CB", "CB"):
            continue
        if point_in_obb(pc, corners):
            boxes.append(det["box"])
    return boxes


def _min_dist(p, centers):
    if not centers:
        return math.inf
    return min(math.hypot(p[0] - c[0], p[1] - c[1]) for c in centers)


def _sweep_angles(base, lo, hi, step):
    yield base
    d = step
    while d <= max(hi - base, base - lo):
        if base + d <= hi:
            yield base + d
        if base - d >= lo:
            yield base - d
        d += step


def predict_next(per_card_path, seat_name, is_double=False,
                 default_step=(40.0, 0.0), cards_dir=DEFAULT_CARDS_DIR):
    data = json.load(open(per_card_path))
    sid = SEAT_ID[seat_name]
    centers, ts = _seat_centers_in_order(data, sid)
    if not centers:
        raise SystemExit(f"No cards assigned to {seat_name}")
    if is_double and len(centers) != 2:
        raise SystemExit(
            f"--double requires exactly 2 cards at {seat_name}; found {len(centers)}."
        )
    seat_boxes = _seat_boxes_at_rep(data, cards_dir, sid)
    if seat_boxes:
        big_w = max(b[2] - b[0] for b in seat_boxes)
        big_h = max(b[3] - b[1] for b in seat_boxes)
    else:
        big_w, big_h = 40.0, 60.0
    last = centers[-1]
    if len(centers) >= 2:
        step = (centers[1][0] - centers[0][0], centers[1][1] - centers[0][1])
    else:
        step = default_step
    avoid = [c for c in _all_centers(data)
             if math.hypot(c[0] - last[0], c[1] - last[1]) > OVERLAP_RADIUS_PX]

    if is_double:
        base_ang = DOUBLE_ANGLE_DEG
        lo, hi = DOUBLE_ANGLE_BOUNDS
    else:
        base_ang = HIT_ANGLE_DEG
        lo, hi = HIT_ANGLE_DEG - NORMAL_MAX_TWEAK_DEG, HIT_ANGLE_DEG + NORMAL_MAX_TWEAK_DEG

    chosen_ang, nxt_step, nxt = None, None, None
    for ang in _sweep_angles(base_ang, lo, hi, ANGLE_STEP_DEG):
        ns = _rotate(step, ang)
        cand = (last[0] + ns[0], last[1] + ns[1])
        if _min_dist(cand, avoid) >= OVERLAP_RADIUS_PX:
            chosen_ang, nxt_step, nxt = ang, ns, cand
            break
    if nxt is None:
        chosen_ang = base_ang
        nxt_step = _rotate(step, base_ang)
        nxt = (last[0] + nxt_step[0], last[1] + nxt_step[1])

    return {
        "seat": seat_name,
        "n_cards": len(centers),
        "ordinals": [(t["ordinal"], t["rank"], t["suit"]) for t in ts],
        "last_center": last,
        "fan_step": step,
        "next_step": nxt_step,
        "chosen_angle_deg": chosen_ang,
        "is_double": is_double,
        "predicted_next": nxt,
        "bigger_bbox": (big_w, big_h),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per_card", required=True)
    p.add_argument("--seat", required=True, choices=list(SEAT_ID))
    p.add_argument("--double", action="store_true",
                   help="Predict assuming the next card is a double.")
    p.add_argument("--cards_dir", default=DEFAULT_CARDS_DIR,
                   help="Where to find {round}_card.jsonl for raw bbox lookup.")
    p.add_argument("--vis_out", default=None,
                   help="If set, draw existing cards + prediction on the round's "
                        "_vis.jpg and write to this path.")
    args = p.parse_args()

    out = predict_next(args.per_card, args.seat, args.double,
                       cards_dir=args.cards_dir)
    print(f"Seat: {out['seat']}  ({out['n_cards']} card(s) so far)")
    for ordi, r, s in out["ordinals"]:
        print(f"  ord {ordi}: {r}{s}")
    print(f"Last center : ({out['last_center'][0]:.1f}, {out['last_center'][1]:.1f})")
    print(f"Fan step    : ({out['fan_step'][0]:.1f}, {out['fan_step'][1]:.1f})")
    print(f"Predicted next ({'double' if out['is_double'] else 'normal'}): "
          f"({out['predicted_next'][0]:.1f}, {out['predicted_next'][1]:.1f})  "
          f"angle={out['chosen_angle_deg']:.1f} deg from "
          f"{'DOUBLE_ANGLE_DEG' if out['is_double'] else 'fan_step'}")
    print(f"Bigger bbox at this seat: w={out['bigger_bbox'][0]:.1f}  "
          f"h={out['bigger_bbox'][1]:.1f}")

    if args.vis_out:
        import cv2
        vis_src = str(Path(args.per_card).with_name(
            Path(args.per_card).name.replace("_per_card.json", "_vis.jpg")))
        img = cv2.imread(vis_src)
        if img is None:
            raise SystemExit(f"Couldn't load {vis_src}")
        lx, ly = int(out["last_center"][0]), int(out["last_center"][1])
        nx, ny = int(out["predicted_next"][0]), int(out["predicted_next"][1])
        bw, bh = out["bigger_bbox"]
        col = (255, 255, 255) if not out["is_double"] else (0, 165, 255)
        cv2.circle(img, (lx, ly), 8, (255, 255, 255), 2)
        cv2.arrowedLine(img, (lx, ly), (nx, ny), col, 3, tipLength=0.25)
        x1, y1 = int(nx - bw / 2), int(ny - bh / 2)
        x2, y2 = int(nx + bw / 2), int(ny + bh / 2)
        cv2.rectangle(img, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (0, 0, 0), 5)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 3)
        cv2.circle(img, (nx, ny), 4, col, -1)
        label = f"{out['seat']} next ({'D' if out['is_double'] else 'N'})"
        cv2.putText(img, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(img, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        cv2.imwrite(args.vis_out, img)
        print(f"Wrote {args.vis_out}")


if __name__ == "__main__":
    main()
