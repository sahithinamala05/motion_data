"""Red-card + short-lived filter for initial-hand card-layout predictions.

Operates on the per-round initial-hand JSONs (output of compute_initial_hand_cards.py:
each `initial_hands` entry has label / start_frame / end_frame / card_layout). Drops a
motion (a single "initial hands 1st"/"2nd" segment, keeping its counterpart) when either:

  RED-CARD (heuristic, no video needed): the segment's card_layout contains a stable card
    that, in the 1280x720 video frame, sits at x<350 AND y<500 AND is >90px from every
    PLAYER seat (seats 1-7). This is the shoe's red/cut card resting to the left of the
    players. Card-layout centroids are in 1920x1080, so we scale by /1.5 to the 1280x720
    space the thresholds are defined in. Carry-over aware: a card lingering from the
    previous segment (within 40px) does NOT re-flag the later segment — only the segment
    where the red card FIRST appears is dropped.

  SHORT-LIVED: segment duration < MIN_DUR frames (spurious 1-few-frame detections).

Verified: the red heuristic fires on the genuine cut card (e.g. card-layout (504,686) ->
video (336,457)) and NOT on red suits/ranks (those sit on bright white cards) or the
dealer's own cards near the shoe (x>>350). Video-based color detection was rejected because
only ~7,935 of the 25k roundcuts have a video on disk; this heuristic runs on all of them.

Usage:  python redcard_filter.py --src <layout_dir> --dst <out_dir> [--min_dur 30]
"""
import os, json, argparse
import numpy as np

# player seats 1-7 (rightmost..leftmost) in 1920x1080
PLAYERS = np.array([[1389,777],[1257,829],[1125,861],[963,892],[796,879],[688,837],[562,778]], float)
IH = ("initial hands 1st", "initial hands 2nd")
X_MAX_1280, Y_MAX_1280 = 350, 500      # red-card region in the 1280x720 video frame
PLAYER_MIN_DIST = 90                    # px (1920 space) a red card must be from every player seat
CARRY_TOL = 40                         # px: same card lingering across segments

def redcard_points(seg):
    """Centroids in this segment's card_layout that satisfy the red-card heuristic."""
    pts = []
    for run in seg.get("card_layout", []):
        for c in run.get("cards", []):
            x, y = c["centroid"]                        # 1920x1080
            if x/1.5 < X_MAX_1280 and y/1.5 < Y_MAX_1280 \
               and np.min(np.linalg.norm(PLAYERS - np.array([x, y]), axis=1)) > PLAYER_MIN_DIST:
                pts.append((x, y))
    return pts

def filter_clip(d, min_dur):
    """Return (new_dict, dropped_short, dropped_red, red_labels)."""
    segs = sorted([e for e in d.get("initial_hands", []) if e.get("label") in IH],
                  key=lambda e: e.get("start_frame", 0))
    other = [e for e in d.get("initial_hands", []) if e.get("label") not in IH]
    seen = []; kept = []; dshort = dred = 0; red_labels = []
    for e in segs:
        dur = e["end_frame"] - e["start_frame"] + 1
        pts = redcard_points(e)
        new = [p for p in pts if all(np.hypot(p[0]-q[0], p[1]-q[1]) > CARRY_TOL for q in seen)]
        seen.extend(pts)
        if dur < min_dur:
            dshort += 1; continue
        if new:
            dred += 1; red_labels.append(e["label"]); continue
        kept.append(e)
    d = dict(d); d["initial_hands"] = kept + other
    return d, dshort, dred, red_labels

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="initial-hand card-layout dir (*_predictions.json)")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--min_dur", type=int, default=30)
    ap.add_argument("--flags", default="")
    a = ap.parse_args()
    os.makedirs(a.dst, exist_ok=True)
    files = [f for f in os.listdir(a.src) if f.endswith("_predictions.json")]
    tot_short = tot_red = wrote = emptied = 0; flag_lines = []
    for f in files:
        cid = f[:-len("_predictions.json")]
        d = json.load(open(os.path.join(a.src, f)))
        had = any(e.get("label") in IH for e in d.get("initial_hands", []))
        nd, ds, dr, labs = filter_clip(d, a.min_dur)
        tot_short += ds; tot_red += dr
        for lab in labs: flag_lines.append(f"{cid}\t{lab}")
        if had and not any(e.get("label") in IH for e in nd["initial_hands"]): emptied += 1
        json.dump(nd, open(os.path.join(a.dst, f), "w")); wrote += 1
    if a.flags: open(a.flags, "w").write("\n".join(flag_lines) + "\n")
    print(f"wrote {wrote} -> {a.dst}")
    print(f"dropped short(<{a.min_dur}f) {tot_short} | dropped red-card {tot_red} | rounds emptied {emptied}")
