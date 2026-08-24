"""Add per-card start_point/end_point and ordered start_points/end_points ([card0, card1])
to existing SRSS SAM trajectory JSONLs, derived from each card's trajectory first/last frame.
Cards are kept in their existing card0/card1 order (no reordering)."""
import glob, json, os

TRAJ_DIR = "/home/ubuntu/us-west-3-fs/sahithi/hard_action_annotations/full_run/split_discard_gated_srss/sam_tracking/trajectories"


def main():
    files = sorted(glob.glob(os.path.join(TRAJ_DIR, "*_srss_trajectory.jsonl")))
    print(f"adding start/end points to {len(files)} files ...")
    n = 0
    for p in files:
        recs = [json.loads(l) for l in open(p) if l.strip()]
        for r in recs:
            for c in r.get("cards", []):
                tj = c.get("trajectory") or []
                if tj:
                    c["start_point"] = {"frame": tj[0][0], "cx": tj[0][1], "cy": tj[0][2]}
                    c["end_point"] = {"frame": tj[-1][0], "cx": tj[-1][1], "cy": tj[-1][2]}
                else:
                    c["start_point"] = c["end_point"] = None
            r["start_points"] = [[c["start_point"]["cx"], c["start_point"]["cy"]] if c.get("start_point") else None
                                 for c in r.get("cards", [])]
            r["end_points"] = [[c["end_point"]["cx"], c["end_point"]["cy"]] if c.get("end_point") else None
                               for c in r.get("cards", [])]
            n += 1
        with open(p, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
    print(f"done: {len(files)} files, {n} records updated with start_points/end_points (card0,card1 order)")


if __name__ == "__main__":
    main()
