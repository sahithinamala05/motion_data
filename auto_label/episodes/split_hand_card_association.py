"""
Associate dealer hands with split cards using dwpose tracking.

For same-suit pairs where card identity is ambiguous, uses dealer hand
positions to determine which hand deals with which card.

Logic:
  1. Load dwpose pkl for the video
  2. At the start frame, find which hand is closest to which card
  3. At the end frame, use the same hand-card mapping to identify cards
  4. The card near the same hand at start and end = same physical card

Usage:
    from split_hand_card_association import resolve_same_suit_pair

    e0, e1 = resolve_same_suit_pair(
        dwpose_data, start_frame, end_frame,
        start_cards=(s0, s1),   # sorted: s0=upper, s1=lower
        end_cards=(ea, eb),     # unsorted end detections
        frame_dims=(1280, 720),
    )
"""

import math
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np

# ─── Constants ───────────────────────────────────────────────────────────────
WRIST_KP = 0           # hand keypoint index for wrist
INDEX_TIP_KP = 8       # hand keypoint index for index finger tip
HAND_CONF_THRESH = 0.1 # minimum average keypoint value to consider hand detected

# Compatibility shim for numpy 2.0+ pickles
if not hasattr(np, '_core'):
    import numpy.core as _np_core
    import sys as _sys
    _sys.modules.setdefault('numpy._core', _np_core)
    for _name in dir(_np_core):
        _sys.modules.setdefault(f'numpy._core.{_name}', getattr(_np_core, _name))


# ─── Helpers ─────────────────────────────────────────────────────────────────

def load_dwpose(pkl_path: str) -> Optional[list]:
    """Load dwpose pkl file. Returns list of per-frame dicts or None."""
    try:
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None


def _get_hand_position(frame_data: dict, hand_idx: int,
                       frame_w: int, frame_h: int) -> Optional[Tuple[float, float]]:
    """Get hand position in pixel coordinates.
    Uses index finger tip (kp 8), falls back to wrist (kp 0).
    Returns (x, y) or None if hand not detected."""
    hands = frame_data["pose"]["hands"]  # (2, 21, 2) normalized
    hand = np.array(hands[hand_idx])

    # check if hand is detected (not all zeros)
    if hand.mean() < HAND_CONF_THRESH:
        return None

    # prefer index finger tip, fallback to wrist
    for kp in [INDEX_TIP_KP, WRIST_KP]:
        x_norm, y_norm = hand[kp]
        if x_norm > 0.01 and y_norm > 0.01:
            return (x_norm * frame_w, y_norm * frame_h)

    return None


def _dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _card_center(card: dict) -> Tuple[float, float]:
    c = card["polygon_center"][0]
    return (c[0], c[1])


# ─── Main logic ──────────────────────────────────────────────────────────────

def get_hand_card_mapping(
    dwpose_data: list,
    frame_idx: int,
    card_a: dict,
    card_b: dict,
    frame_w: int = 1280,
    frame_h: int = 720,
) -> Optional[Dict[int, int]]:
    """At a given frame, determine which hand is closest to which card.

    Returns {hand_idx: card_idx} e.g. {0: 0, 1: 1} meaning hand 0 is near
    card 0 (a) and hand 1 is near card 1 (b). Returns None if hands not detected.
    """
    if frame_idx >= len(dwpose_data):
        return None

    fd = dwpose_data[frame_idx]
    hand0_pos = _get_hand_position(fd, 0, frame_w, frame_h)
    hand1_pos = _get_hand_position(fd, 1, frame_w, frame_h)

    if hand0_pos is None or hand1_pos is None:
        return None

    ca = _card_center(card_a)
    cb = _card_center(card_b)

    # compute all distances
    d_h0_ca = _dist(hand0_pos, ca)
    d_h0_cb = _dist(hand0_pos, cb)
    d_h1_ca = _dist(hand1_pos, ca)
    d_h1_cb = _dist(hand1_pos, cb)

    # greedy assignment: assign each hand to its nearest card
    if d_h0_ca + d_h1_cb <= d_h0_cb + d_h1_ca:
        return {0: 0, 1: 1}  # hand0→card_a, hand1→card_b
    else:
        return {0: 1, 1: 0}  # hand0→card_b, hand1→card_a


def resolve_same_suit_pair(
    dwpose_data: list,
    start_frame_idx: int,
    end_frame_idx: int,
    start_cards: Tuple[dict, dict],
    end_cards: Tuple[dict, dict],
    frame_w: int = 1280,
    frame_h: int = 720,
) -> Tuple[dict, dict]:
    """For same-suit pairs, use hand positions to match end cards to start cards.

    Args:
        dwpose_data: loaded dwpose pkl (list of per-frame dicts)
        start_frame_idx: frame index into dwpose_data for start detection
        end_frame_idx: frame index into dwpose_data for end detection
        start_cards: (s0, s1) sorted cards at start (s0=upper)
        end_cards: (ea, eb) unsorted end detections

    Returns:
        (e0, e1) where e0 is the same physical card as s0, e1 same as s1
    """
    s0, s1 = start_cards
    ea, eb = end_cards

    # get hand-card mapping at start
    start_map = get_hand_card_mapping(
        dwpose_data, start_frame_idx, s0, s1, frame_w, frame_h
    )
    if start_map is None:
        # can't determine — fallback to proximity
        return ea, eb

    # get hand-card mapping at end (using ea, eb)
    end_map = get_hand_card_mapping(
        dwpose_data, end_frame_idx, ea, eb, frame_w, frame_h
    )
    if end_map is None:
        return ea, eb

    # find which hand was near s0 at start
    hand_for_s0 = None
    for hand_idx, card_idx in start_map.items():
        if card_idx == 0:  # card_a = s0
            hand_for_s0 = hand_idx
            break

    if hand_for_s0 is None:
        return ea, eb

    # at end, which card is near that same hand?
    end_card_for_hand = end_map.get(hand_for_s0)

    if end_card_for_hand == 0:
        # hand_for_s0 is near ea → ea=e0, eb=e1
        return ea, eb
    else:
        # hand_for_s0 is near eb → eb=e0, ea=e1
        return eb, ea


def resolve_pair_identity(
    dwpose_data: Optional[list],
    start_frame_idx: int,
    end_frame_idx: int,
    start_cards: Tuple[dict, dict],
    end_cards: Tuple[dict, dict],
    swapped: bool,
    frame_w: int = 1280,
    frame_h: int = 720,
) -> Tuple[dict, dict]:
    """Main entry point. Resolves end card order to match start card identity.

    For different-suit pairs: uses suit matching (reliable).
    For same-suit pairs: uses dwpose hand tracking if available,
                         falls back to swap-based matching.

    Args:
        dwpose_data: loaded dwpose pkl or None
        start_frame_idx: frame index for start detection
        end_frame_idx: frame index for end detection
        start_cards: (s0, s1) sorted by y (s0=upper)
        end_cards: (ea, eb) from match_pair_at_frame
        swapped: whether sa/sb were swapped to get s0/s1

    Returns:
        (e0, e1) matched to (s0, s1)
    """
    s0, s1 = start_cards
    ea, eb = end_cards

    s0_suit = s0.get("suit", "")
    s1_suit = s1.get("suit", "")

    # different suits → match by suit (reliable)
    if s0_suit and s1_suit and s0_suit != s1_suit and s0_suit != "U" and s1_suit != "U":
        ea_suit = ea.get("suit", "")
        if ea_suit == s0_suit:
            return ea, eb
        else:
            return eb, ea

    # same suit → try dwpose
    if dwpose_data is not None:
        e0, e1 = resolve_same_suit_pair(
            dwpose_data, start_frame_idx, end_frame_idx,
            start_cards, end_cards, frame_w, frame_h,
        )
        return e0, e1

    # no dwpose → fallback to swap-based matching
    if swapped:
        return eb, ea
    return ea, eb
