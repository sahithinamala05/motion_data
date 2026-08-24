"""Extract ONLY the flagged initial-hand segment from the first 100 clips in
beyond_p7_lt500_FULLBATCH.txt and save short clips to initial_hand_red_card_vis.

For each listed clip we read its segment label (e.g. "initial hands 1st"),
look up that segment's local frame range in the initial_hand prediction, and
cut just that span out of the source round video.
"""
import os, re, json, subprocess

LIST   = "/home/ubuntu/us-west-3-fs/sahithi/beyond_p7_lt500_FULLBATCH.txt"
PRED   = "/home/ubuntu/us-west-3-fs/sahithi/initial_hand"
SRC    = "/home/ubuntu/us-west-3-fs/live_dealer_blackjack/roundcut/good_quality_rounds"
OUT    = "/home/ubuntu/us-west-3-fs/sahithi/initial_hand_red_card_vis"
N      = 100
os.makedirs(OUT, exist_ok=True)

def parse_lines():
    rows = []
    with open(LIST) as f:
        for ln in f:
            parts = ln.split()
            if len(parts) < 4:        # header / separator lines
                continue
            if parts[1].count("_") < 3:   # not a clip stem
                continue
            stem = parts[1]
            seg  = " ".join(parts[3:])    # e.g. "initial hands 1st"
            rows.append((stem, seg))
    return rows

def fps_of(path):
    out = subprocess.run(
        ["ffprobe","-v","0","-select_streams","v:0","-show_entries",
         "stream=r_frame_rate","-of","default=nk=1:nw=1", path],
        capture_output=True, text=True).stdout.strip()
    num, den = out.split("/") if "/" in out else (out, "1")
    return float(num)/float(den) if float(den) else 30.0

def seg_range(stem, seg):
    p = os.path.join(PRED, f"{stem}_predictions.json")
    if not os.path.exists(p): return None
    d = json.load(open(p))
    rng = [(ih["start_frame"], ih["end_frame"]) for ih in d.get("initial_hands", [])
           if ih.get("label") == seg]
    if not rng: return None
    return min(s for s,_ in rng), max(e for _,e in rng)   # union if 2 hands

rows = parse_lines()[:N]
print(f"[init] {len(rows)} clips to process -> {OUT}", flush=True)
ok = skip = fail = 0
for i, (stem, seg) in enumerate(rows, 1):
    src = os.path.join(SRC, f"{stem}.mp4")
    tag = seg.replace(" ", "_")
    r = seg_range(stem, seg)
    if not os.path.exists(src) or r is None:
        fail += 1
        print(f"[{i}/{len(rows)}] MISS {stem} (src={os.path.exists(src)} range={r})", flush=True)
        continue
    s_f, e_f = r
    out = os.path.join(OUT, f"{stem}__{tag}__{s_f}-{e_f}.mp4")
    if os.path.exists(out) and os.path.getsize(out) > 0:
        skip += 1; continue
    fps = fps_of(src)
    ss = s_f / fps
    dur = (e_f - s_f + 1) / fps
    rc = subprocess.run(
        ["ffmpeg","-y","-loglevel","error","-i",src,"-ss",f"{ss:.3f}","-t",f"{dur:.3f}",
         "-c:v","libx264","-pix_fmt","yuv420p","-movflags","+faststart",out],
        capture_output=True, text=True)
    if rc.returncode == 0 and os.path.exists(out) and os.path.getsize(out) > 0:
        ok += 1
        print(f"[{i}/{len(rows)}] ok {stem} [{seg}] {s_f}-{e_f}", flush=True)
    else:
        fail += 1
        print(f"[{i}/{len(rows)}] FFMPEG_FAIL {stem}: {rc.stderr[:200]}", flush=True)

print(f"\n[done] ok={ok} skip={skip} fail={fail} -> {OUT}", flush=True)
