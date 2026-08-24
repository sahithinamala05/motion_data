"""Re-verify saved trajectories against the strict split_verdict (both cards must move),
without re-running SAM. Trajectories that no longer pass are MOVED to a `_nonsplit/`
subdir (kept for review), so `trajectories/` holds only clearly-valid splits.
Recomputes the verdict from the saved per-card trajectory points.
"""
import glob, json, os, shutil, sys

TRAJ_DIR = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/ubuntu/us-west-3-fs/sahithi/hard_action_annotations/full_run/split_sam_traj_all/trajectories"
NONSPLIT_DIR = os.path.join(os.path.dirname(TRAJ_DIR), "trajectories_nonsplit")
os.makedirs(NONSPLIT_DIR, exist_ok=True)

# thresholds (must match sam_track_srss.py)
SPLIT_MIN_SEP = 35.0
SPLIT_MIN_TRAVEL = 15.0
SPLIT_MIN_SPREAD = 20.0
BOTH_MIN_TRAVEL = 12.0
SPLIT_MAX_BOX_FRAC = 0.04


def _path_len(traj):
    return sum(((traj[i][1] - traj[i-1][1])**2 + (traj[i][2] - traj[i-1][2])**2) ** 0.5
               for i in range(1, len(traj)))


def verdict(cards):
    if len(cards) < 2:
        return False, "fewer than 2 cards tracked"
    a = {t[0]: (t[1], t[2]) for t in cards[0]["trajectory"]}
    b = {t[0]: (t[1], t[2]) for t in cards[1]["trajectory"]}
    common = sorted(set(a) & set(b))
    if not common:
        return False, "no overlapping frames"
    seps = [((a[f][0]-b[f][0])**2 + (a[f][1]-b[f][1])**2) ** 0.5 for f in common]
    sep_max, sep_min = max(seps), min(seps)
    spread = sep_max - sep_min
    travels = [_path_len(c["trajectory"]) for c in cards]
    max_travel, min_travel = max(travels), min(travels)
    max_box = max((c.get("max_box_frac") or 0.0) for c in cards)
    if max_box > SPLIT_MAX_BOX_FRAC:
        return False, "track blew up"
    if max_travel < SPLIT_MIN_TRAVEL and spread < SPLIT_MIN_SPREAD:
        return False, "cards static"
    if min_travel < BOTH_MIN_TRAVEL:
        return False, "one card static (wrong/mis-paired)"
    if sep_max < SPLIT_MIN_SEP:
        return False, "cards never separate"
    return True, "ok"


def main():
    files = sorted(glob.glob(os.path.join(TRAJ_DIR, "*.jsonl")))
    kept = moved = 0
    reasons = {}
    for p in files:
        recs = [json.loads(l) for l in open(p) if l.strip()]
        # a file passes if ANY of its segment records is a valid split
        any_valid = False
        for r in recs:
            ok, why = verdict(r.get("cards", []))
            r["is_split"] = ok
            r["split_stats"] = {**r.get("split_stats", {}), "reason": why}
            if ok:
                any_valid = True
            else:
                reasons[why] = reasons.get(why, 0) + 1
        if any_valid:
            with open(p, "w") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
            kept += 1
        else:
            shutil.move(p, os.path.join(NONSPLIT_DIR, os.path.basename(p)))
            moved += 1
    print(f"kept (valid split): {kept} | moved to non-split: {moved}")
    print("drop reasons:", reasons)
    print(f"non-split trajectories: {NONSPLIT_DIR}")


if __name__ == "__main__":
    main()
