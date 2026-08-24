"""Reorder cards in existing SRSS SAM trajectory JSONLs to the red/blue convention:
index 0 = upper card (smaller y) at the first shared frame ('red'),
index 1 = lower card ('blue'). Adds a 'color' field. Rewrites files in place.
"""
import glob, json, os

TRAJ_DIR = "/home/ubuntu/us-west-3-fs/sahithi/hard_action_annotations/full_run/split_discard_gated_srss/sam_tracking/trajectories"


def order_cards(cards):
    if len(cards) >= 2:
        a = {t[0]: t[2] for t in cards[0]["trajectory"]}
        b = {t[0]: t[2] for t in cards[1]["trajectory"]}
        common = sorted(set(a) & set(b))
        if common:
            ya, yb = a[common[0]], b[common[0]]
        else:
            ya = cards[0]["trajectory"][0][2] if cards[0]["trajectory"] else 0
            yb = cards[1]["trajectory"][0][2] if cards[1]["trajectory"] else 0
        if ya > yb:
            cards = [cards[1], cards[0]]
    for i, c in enumerate(cards):
        c["card_index"] = i
        c["color"] = "red" if i == 0 else ("blue" if i == 1 else "other")
    return cards


def main():
    files = sorted(glob.glob(os.path.join(TRAJ_DIR, "*_srss_trajectory.jsonl")))
    print(f"reordering {len(files)} trajectory files ...")
    n_recs = 0
    for p in files:
        recs = [json.loads(l) for l in open(p) if l.strip()]
        for r in recs:
            r["cards"] = order_cards(r.get("cards", []))
            n_recs += 1
        with open(p, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
    print(f"done: {len(files)} files, {n_recs} records reordered (index0=red/upper, index1=blue/lower)")


if __name__ == "__main__":
    main()
