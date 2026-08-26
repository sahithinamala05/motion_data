"""Populate meta_text + xy_seat on initial-hand card-layout JSONs (all_actions schema).

For each "initial hands 1st"/"2nd" segment: from the settled card_layout run (max n_cards),
take one card per seat (the max-`order` card = the relevant card for that motion) and emit:
  meta_text : "<player seats ascending><dealer 0 if a seat-0 card exists>"  e.g. "2345670"
  xy_seat   : "(x, y) (x, y) ..."  one coord per seat, SAME order as meta_text, 2 decimals

Matches all_actions' convention: players 1-7 ascending, dealer (0) appended LAST, and only
when a dealer card is actually detected (detection-based; the dealer up-card is often not yet
settled during ih1, so ih1 can legitimately lack 0 while ih2 has it).

Skips clips whose stems are in --skip (e.g. the 378 human-adjusted, which keep their own values).
Usage:  python populate_meta_xy.py --dir <layout_dir> [--skip <human_dir>]
"""
import os, json, argparse

IH = ("initial hands 1st", "initial hands 2nd")

def meta_xy(seg):
    runs = seg.get("card_layout", [])
    if not runs:
        return None, None
    r = max(runs, key=lambda r: r.get("n_cards", 0))
    byseat = {}
    for c in r.get("cards", []):
        s = c.get("seat"); o = c.get("order", 0)
        if s is None:
            continue
        if s not in byseat or o > byseat[s][0]:
            byseat[s] = (o, c["centroid"])
    if not byseat:
        return "", ""
    players = sorted(s for s in byseat if s != 0)
    order = players + ([0] if 0 in byseat else [])          # dealer 0 last, only if detected
    meta = "".join(str(s) for s in order)
    xy = " ".join(f"({byseat[s][1][0]:.2f}, {byseat[s][1][1]:.2f})" for s in order)
    return meta, xy

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="initial-hand card-layout dir (*_predictions.json)")
    ap.add_argument("--skip", default="", help="dir of clips to leave untouched (e.g. human-adjusted)")
    a = ap.parse_args()
    skip = set()
    if a.skip and os.path.isdir(a.skip):
        skip = set(f[:-len("_predictions.json")] for f in os.listdir(a.skip) if f.endswith("_predictions.json"))
    files = [f for f in os.listdir(a.dir) if f.endswith("_predictions.json")]
    upd = segs = skipped = 0
    for f in files:
        cid = f[:-len("_predictions.json")]
        if cid in skip:
            skipped += 1; continue
        p = os.path.join(a.dir, f); d = json.load(open(p)); changed = False
        for e in d.get("initial_hands", []):
            if e.get("label") not in IH:
                continue
            m, xy = meta_xy(e)
            if m is None:
                continue
            e["meta_text"] = [m]; e["xy_seat"] = [xy]; segs += 1; changed = True
        if changed:
            json.dump(d, open(p, "w")); upd += 1
    print(f"populated: files {upd} | ih segments {segs} | skipped(human) {skipped}")
