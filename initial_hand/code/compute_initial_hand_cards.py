#!/usr/bin/env python3
"""
Compute per-card layout for the "initial hands" segments of each prediction JSON.

For each prediction JSON in the annotation directory:
1. Find every "initial hands 1st" / "initial hands 2nd" segment.
2. Look up the card detections at the segment's LAST frame (same global-frame
   math as compute_meta_text_and_export_v2.py).
3. Cluster detections to the 8 template seats (0 = dealer, 1-7 = players) using
   clustering_util.assign_cards_to_positions.
4. Emit an ordered list of cards (sorted left-to-right by x), each annotated with
   its centroid, role (dealer/player), seat, and order index.
5. Store the list on the segment as "card_layout" and save the updated JSON to
   the output directory.

The dealer is identified by clustering seat 0 (the template puts the dealer at the
top-center of the table, y ~ 723 in 1920x1080); player cards sit lower in an arc.
"""

import os
import sys
import json
import glob
from collections import defaultdict, Counter

from tqdm import tqdm

# This script was moved here from auto_label/10s-chunk/. Its shared helpers
# (clustering_util, compute_meta_text_and_export_v2) stay in that directory
# because ~8 other scripts import them, so add it to the path for the bare
# imports below.
_SHARED = "/home/ubuntu/sahithi/motion-data-process/auto_label/10s-chunk"
if _SHARED not in sys.path:
    sys.path.insert(0, _SHARED)

from clustering_util import assign_cards_to_positions, N_POSITIONS

# Reuse the file-parsing / card-lookup helpers from v2 so the frame math stays
# identical between the two scripts.
from compute_meta_text_and_export_v2 import (
    parse_pred_json_filename,
    group_card_detection_files,
    find_card_files_for_frame_range,
)


# ==================== CONFIGURATION ====================
PRED_JSON_DIR = (
    "/home/ubuntu/us-west-3-fs/live_dealer_blackjack/action_annotation/"
    "mixed_mini_batch/annotation/AutoLabeling_batch_01_part_1"
)
CARD_DETECTION_DIR = (
    "/home/ubuntu/us-west-3-fs/live_dealer_blackjack/yolo_cards/card_detection/all_jsons"
)
OUTPUT_DIR = "/home/ubuntu/us-west-3-fs/sahithi/initial_hand"

INITIAL_HAND_LABELS = {"initial hands 1st", "initial hands 2nd"}

# A per-seat configuration must persist at least this many consecutive frames to
# start a new run; shorter blips (detection dropouts / cards in transit) are
# absorbed into the surrounding stable run.
MIN_STABLE_FRAMES = 3

# Minimum detection confidence to count a card as "settled" on the table. Genuine
# placed cards score ~0.89+ (players ~0.95); in-transit cards mis-clustered to the
# dealer score much lower (e.g. 0.30), so this drops that noise without losing real
# cards. Verified: settled dealer cards are all >= 0.889 in this batch.
CONFIDENCE_THRESHOLD = 0.8


# ==================== CARD DETECTION LOOKUP (cached) ====================

# Cache parsed card JSONs. Cleared per prediction file to bound memory, since a
# file's segments only touch the handful of chunks covering their frame range.
_CARD_JSON_CACHE = {}


def _load_card_json(filepath):
    data = _CARD_JSON_CACHE.get(filepath)
    if data is None:
        with open(filepath) as f:
            data = json.load(f)
        _CARD_JSON_CACHE[filepath] = data
    return data


def detections_for_global_frame(card_files, gframe):
    """Return the card detections at a single global frame, or []."""
    for filepath, cstart, _cend in find_card_files_for_frame_range(card_files, gframe, gframe):
        data = _load_card_json(filepath)
        rel = gframe - cstart
        for entry in data:
            if entry.get("frame_id") == rel:
                return entry.get("detections", [])
        if 0 <= rel < len(data):
            return data[rel].get("detections", [])
    return []


# ==================== CORE LOGIC ====================

def build_card_layout(detections):
    """
    Cluster detections to seats and return a card list in DEALING order.

    Dealing proceeds player 1 -> player 7 (seat 1 = rightmost ... seat 7 = leftmost),
    and the dealer takes their card after the players, so the dealer (seat 0) is
    ordered last. Within a seat, cards are tiebroken by x. Each entry:
        {"order": int, "centroid": [x, y], "role": "dealer"|"player", "seat": 0-7}
    Only cards assigned to a seat (dealer or players 1-7) are included.
    """
    detections = [d for d in detections if d.get("confidence", 1.0) >= CONFIDENCE_THRESHOLD]
    clusters = assign_cards_to_positions(detections)

    cards = []
    for seat in range(N_POSITIONS):  # 0 = dealer, 1-7 = players
        for det in clusters[seat]:
            cx, cy = det["polygon_center"]
            cards.append(
                {
                    "centroid": [round(float(cx), 2), round(float(cy), 2)],
                    "role": "dealer" if seat == 0 else "player",
                    "seat": seat,
                    "confidence": det.get("confidence"),
                }
            )

    # Sort by seat (players 1..7 then dealer); dealing `order` is assigned per-run
    # by assign_dealing_order once the run's settled card set is known.
    cards.sort(key=lambda c: (c["seat"] == 0, c["seat"], c["centroid"][0]))
    return cards


def assign_dealing_order(cards):
    """Assign the standard blackjack dealing `order` to a settled card set, in place.

    Dealing protocol, independent of (noisy) detection timing: within each round the
    active players are dealt in seat order (1=rightmost .. 7=leftmost), then the dealer
    (seat 0) last. A seat's j-th card (ordered by x) belongs to round j, so:
        order = j * (#seats dealt this round) + dealing_rank(seat)
    -> round 1 = players then dealer, round 2 = players then dealer, contiguous 0..N-1.
    The first card is therefore always the first active player; the dealer is never 0.
    """
    seats_present = sorted({c["seat"] for c in cards}, key=lambda s: (s == 0, s))
    rank = {s: i for i, s in enumerate(seats_present)}
    per_round = len(seats_present)

    by_seat = defaultdict(list)
    for c in cards:
        by_seat[c["seat"]].append(c)
    for seat, seat_cards in by_seat.items():
        seat_cards.sort(key=lambda c: c["centroid"][0])
        for j, c in enumerate(seat_cards):
            c["order"] = j * per_round + rank[seat]

    cards.sort(key=lambda c: c["order"])


def _dist2(p, q):
    return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2


def _assign_to_anchors(cards, anchors):
    """One-to-one assign each card to its globally-nearest free anchor.

    Returns {card_index -> anchor_index}. Caller guarantees len(cards) <= len(anchors)
    (anchors are taken from the seat's max-count run), so every card gets an anchor.
    """
    pairs = sorted(
        (_dist2(c["centroid"], a), ci, ai)
        for ci, c in enumerate(cards)
        for ai, a in enumerate(anchors)
    )
    used_c, used_a, res = set(), set(), {}
    for _d, ci, ai in pairs:
        if ci in used_c or ai in used_a:
            continue
        res[ci] = ai
        used_c.add(ci)
        used_a.add(ai)
    return res


def stabilize_run_orders(runs):
    """Number cards by the order they are dealt in the video, fixed once per card.

    `order` is a single contiguous sequence 0..N-1 over the whole hand, following the
    actual placement order observed in the clip; each physical card keeps its number in
    every later frame (no swapping while the next card is being dealt). Specifically:
      1. Per seat, the resting positions (anchors) are the card centroids in the latest
         run where the seat holds its maximum card count (the fully-dealt layout).
      2. Each anchor's first-appearance time is the first run in which a card maps to it.
      3. All anchors across all seats are sorted by (first-appearance time, then dealing
         tiebreak = players seat 1->7 before the dealer, then x) and numbered 0,1,2,...
         -- i.e. the card placed first in the video gets the lowest number.
      4. In every run, each detected card is assigned one-to-one to its nearest anchor
         and inherits that anchor's global number.
    A card sitting on its resting position is ~0 distance from its anchor, so it always
    wins that anchor and keeps its number; an in-transit card is forced onto the
    remaining anchor. Hence the number maps to the same physical card across frames.
    """
    runs_with_cards = [r for r in runs if r["cards"]]
    if not runs_with_cards:
        return

    # 1. Resting anchors per seat = centroids from the LATEST run holding the seat's max
    #    card count (loop forward and overwrite -> ends on the latest, most-settled run).
    max_count = defaultdict(int)
    for run in runs:
        counts = Counter(c["seat"] for c in run["cards"])
        for seat, n in counts.items():
            max_count[seat] = max(max_count[seat], n)
    anchors = {}
    for run in runs:
        by_seat = defaultdict(list)
        for c in run["cards"]:
            by_seat[c["seat"]].append(c)
        for seat, seat_cards in by_seat.items():
            if len(seat_cards) == max_count[seat]:
                anchors[seat] = [c["centroid"] for c in seat_cards]

    # 2. First-appearance run index per anchor (seat, anchor_index).
    first_occ = {(s, ai): None for s, a in anchors.items() for ai in range(len(a))}
    for ridx, run in enumerate(runs):
        by_seat = defaultdict(list)
        for c in run["cards"]:
            by_seat[c["seat"]].append(c)
        for seat, seat_cards in by_seat.items():
            a = anchors.get(seat)
            if not a:
                continue
            for ai in set(_assign_to_anchors(seat_cards, a).values()):
                if first_occ[(seat, ai)] is None:
                    first_occ[(seat, ai)] = ridx

    # 3. One global dealing sequence: sort all anchors by appearance time, then by the
    #    standard tiebreak (players seat order, dealer last) for cards seen the same run.
    def deal_key(key):
        seat, ai = key
        fo = first_occ[key]
        return (fo if fo is not None else 1 << 30, seat == 0, seat, anchors[seat][ai][0])

    global_order = {key: i for i, key in enumerate(sorted(first_occ, key=deal_key))}

    # 4. Assign each run's cards to anchors and emit the fixed global number.
    for run in runs:
        by_seat = defaultdict(list)
        for c in run["cards"]:
            by_seat[c["seat"]].append(c)
        for seat, seat_cards in by_seat.items():
            a = anchors.get(seat)
            if not a:
                for c in seat_cards:
                    c["order"] = 9999
                continue
            assign = _assign_to_anchors(seat_cards, a)
            for ci, c in enumerate(seat_cards):
                c["order"] = global_order.get((seat, assign.get(ci, 0)), 9999)
        run["cards"].sort(key=lambda c: c["order"])


def occupancy_signature(cards):
    """Per-seat card-count signature, used to detect when the layout changes."""
    counts = [0] * N_POSITIONS
    for c in cards:
        counts[c["seat"]] += 1
    return tuple(counts)


def _debounce_signatures(sigs, min_stable):
    """Forward-fill any raw run shorter than min_stable with the previously
    committed signature, so transient blips get absorbed into the stable state."""
    n = len(sigs)
    out = list(sigs)
    prev = None
    i = 0
    while i < n:
        j = i
        while j < n and sigs[j] == sigs[i]:
            j += 1
        if (j - i) >= min_stable or prev is None:
            prev = sigs[i]              # commit this signature
        else:
            for k in range(i, j):       # too short -> absorb into previous state
                out[k] = prev
        i = j
    return out


def compress_to_runs(frames, cards_per_frame, min_stable=MIN_STABLE_FRAMES):
    """Run-length encode frames by per-seat configuration, after debouncing.

    Returns a list of runs: {start_frame, end_frame, n_frames, n_cards, cards}.
    The representative `cards` for a run come from the last frame in the run that
    genuinely had that configuration (settled centroids).
    """
    raw_sigs = [occupancy_signature(c) for c in cards_per_frame]
    sigs = _debounce_signatures(raw_sigs, min_stable)

    runs = []
    n = len(frames)
    i = 0
    while i < n:
        j = i
        while j < n and sigs[j] == sigs[i]:
            j += 1
        rep = j - 1                      # representative frame index
        for k in range(j - 1, i - 1, -1):
            if raw_sigs[k] == sigs[i]:
                rep = k
                break
        runs.append(
            {
                "start_frame": frames[i],
                "end_frame": frames[j - 1],
                "n_frames": j - i,
                "n_cards": len(cards_per_frame[rep]),
                "cards": cards_per_frame[rep],
            }
        )
        i = j
    return runs


# ==================== MAIN ====================

def main():
    print("=" * 70)
    print("COMPUTE INITIAL-HAND CARD LAYOUT FOR PREDICTION JSON FILES")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    card_grouped = group_card_detection_files(CARD_DETECTION_DIR)

    pred_files = sorted(glob.glob(os.path.join(PRED_JSON_DIR, "*_predictions.json")))
    print(f"Prediction JSON files: {len(pred_files)}")

    total_files = 0
    files_with_initial_hands = 0
    files_with_cards = 0
    segment_counts = defaultdict(int)
    segments_with_cards = defaultdict(int)

    for pred_path in tqdm(pred_files, desc="Processing"):
        try:
            base_name, clip_start_1idx, _clip_end_1idx = parse_pred_json_filename(pred_path)
        except Exception as e:
            print(f"\nSkipping {pred_path}: {e}")
            continue

        with open(pred_path, "r") as f:
            pred_data = json.load(f)

        total_files += 1
        card_files = card_grouped.get(base_name, [])
        _CARD_JSON_CACHE.clear()  # bound memory: only this video's chunks stay cached

        had_cards = False
        initial_hand_segments = []

        ih_segments = [
            s for s in pred_data.get("timeline_segments", [])
            if (s.get("labels") or [""])[0] in INITIAL_HAND_LABELS
        ]

        if ih_segments:
            for segment in ih_segments:
                label = segment["labels"][0]
                segment_counts[label] += 1

                global_start = (clip_start_1idx - 1) + segment["start_frame"]
                global_end = (clip_start_1idx - 1) + segment["end_frame"]
                frames = list(range(global_start, global_end + 1))
                cards_per_frame = [
                    build_card_layout(detections_for_global_frame(card_files, gf))
                    for gf in frames
                ]

                seg_has_cards = any(cards_per_frame)
                runs = compress_to_runs(frames, cards_per_frame)

                # Standard blackjack dealing order, fixed per card once placed
                # (no number swapping when a seat's 2nd card appears).
                stabilize_run_orders(runs)

                initial_hand_segments.append(
                    {
                        "label": label,
                        "start_frame": segment.get("start_frame"),
                        "end_frame": segment.get("end_frame"),
                        "global_start_frame": global_start,
                        "global_end_frame": global_end,
                        "meta_text": segment.get("meta_text", []),
                        "card_layout": runs,
                    }
                )

                if seg_has_cards:
                    had_cards = True
                    segments_with_cards[label] += 1

        # Only emit a file for videos that actually contain initial-hands segments.
        if not initial_hand_segments:
            continue

        files_with_initial_hands += 1
        if had_cards:
            files_with_cards += 1

        out_data = {
            "video_name": pred_data.get("video_name", ""),
            "clip_start_1idx": clip_start_1idx,
            "initial_hands": initial_hand_segments,
        }
        out_path = os.path.join(OUTPUT_DIR, os.path.basename(pred_path))
        with open(out_path, "w") as f:
            json.dump(out_data, f, indent=2)

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total files processed:            {total_files}")
    print(f"Files with an initial-hands seg:  {files_with_initial_hands}")
    print(f"Files with >=1 card detected:     {files_with_cards}")
    print("\nSegments seen / with cards, per label:")
    for label in sorted(segment_counts.keys()):
        print(f"  {label}: {segment_counts[label]} seen, {segments_with_cards[label]} with cards")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
