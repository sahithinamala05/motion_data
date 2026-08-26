"""Validate meta_text + xy_seat on initial-hand card-layout JSONs.

Per "initial hands 1st"/"2nd" segment, checks:
  (1) count      : #seats in meta_text == #coords in xy_seat
  (2) order      : players ascending, dealer 0 (if present) LAST
  (3) coords     : i-th xy_seat coord == the i-th seat's card_layout centroid (max-order card)

Reports counts of violations + a few examples. Skips clips in --skip (human-adjusted).
Usage:  python validate_meta_xy.py --dir <layout_dir> [--skip <human_dir>]
"""
import os, json, re, argparse

IH = ("initial hands 1st", "initial hands 2nd")

def settled_byseat(seg):
    runs = seg.get("card_layout", [])
    if not runs:
        return {}
    r = max(runs, key=lambda r: r.get("n_cards", 0)); best = {}
    for c in r.get("cards", []):
        s = c.get("seat"); o = c.get("order", 0)
        if s is None:
            continue
        if s not in best or o > best[s][0]:
            best[s] = (o, tuple(c["centroid"]))
    return {s: xy for s, (o, xy) in best.items()}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--skip", default="")
    ap.add_argument("--tol", type=float, default=0.6, help="px tolerance for coord match (2-dec rounding)")
    a = ap.parse_args()
    skip = set()
    if a.skip and os.path.isdir(a.skip):
        skip = set(f[:-len("_predictions.json")] for f in os.listdir(a.skip) if f.endswith("_predictions.json"))
    tot = cnt_bad = order_bad = coord_bad = empty = skipped = 0
    ex = []
    for f in os.listdir(a.dir):
        if not f.endswith("_predictions.json"):
            continue
        cid = f[:-len("_predictions.json")]
        if cid in skip:
            skipped += 1; continue
        d = json.load(open(os.path.join(a.dir, f)))
        for e in d.get("initial_hands", []):
            if e.get("label") not in IH:
                continue
            tot += 1
            meta = (e.get("meta_text") or [""])[0]; xy = (e.get("xy_seat") or [""])[0]
            seats = [int(ch) for ch in meta]
            coords = [(float(x), float(y)) for x, y in re.findall(r'\(([-\d.]+),\s*([-\d.]+)\)', str(xy))]
            if not seats and not coords:
                empty += 1; continue
            if len(seats) != len(coords):
                cnt_bad += 1
                if len(ex) < 8: ex.append((cid, e["label"], "count", meta, len(coords)))
                continue
            players = [s for s in seats if s != 0]
            if players != sorted(players) or (0 in seats and seats[-1] != 0):
                order_bad += 1
                if len(ex) < 12: ex.append((cid, e["label"], "order", meta))
            byseat = settled_byseat(e)
            for s, co in zip(seats, coords):
                ref = byseat.get(s)
                if ref is None or abs(ref[0] - co[0]) > a.tol or abs(ref[1] - co[1]) > a.tol:
                    coord_bad += 1
                    if len(ex) < 16: ex.append((cid, e["label"], "coord", s, co, ref))
                    break
    print(f"ih segments checked: {tot} | skipped(human) {skipped} | empty {empty}")
    print(f"  (1) count mismatch:            {cnt_bad}")
    print(f"  (2) order violations:          {order_bad}")
    print(f"  (3) coord != card_layout:      {coord_bad}")
    print("PASS" if cnt_bad == order_bad == coord_bad == 0 else "FAIL")
    for e in ex[:8]:
        print("  ", e)
