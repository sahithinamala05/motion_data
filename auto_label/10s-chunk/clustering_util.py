"""
Clustering utilities for assigning card detections to player seat positions.

Template centroids and card detection coordinates are both in 1920x1080 pixel space.
They are normalized to [0, 1] internally before distance computation.
All public functions accept pixel-space detections and handle normalization internally.
"""

from typing import Dict, List

import numpy as np


# ==================== CONSTANTS ====================

IMAGE_W = 1920
IMAGE_H = 1080

# Manually assigned seat centroids, normalized to [0, 1] from a 1920x1080 frame.
# Index 0 = Dealer, indices 1-7 = Player seats (1=rightmost, 7=leftmost).
#
# Original pixel coordinates (1920x1080):
#   Dealer  : [972, 723]
#   Player1 : [1389, 777]
#   Player2 : [1257, 829]
#   Player3 : [1125, 861]
#   Player4 : [963, 892]
#   Player5 : [796, 879]
#   Player6 : [688, 837]
#   Player7 : [562, 778]
MANUAL_TEMPLATE_NORM = np.array(
    [
        [972 / IMAGE_W, 723 / IMAGE_H],   # Dealer  (index 0)
        [1389 / IMAGE_W, 777 / IMAGE_H],  # Player1 (rightmost)
        [1257 / IMAGE_W, 829 / IMAGE_H],  # Player2
        [1125 / IMAGE_W, 861 / IMAGE_H],  # Player3
        [963 / IMAGE_W, 892 / IMAGE_H],   # Player4
        [796 / IMAGE_W, 879 / IMAGE_H],   # Player5
        [688 / IMAGE_W, 837 / IMAGE_H],   # Player6
        [562 / IMAGE_W, 778 / IMAGE_H],   # Player7 (leftmost)
    ],
    dtype=np.float64,
)

N_POSITIONS = 8          # Dealer + 7 player seats
MAX_DISTANCE_NORM = 200 / IMAGE_W  # max distance in normalized space (~0.104)
ANGULAR_WEIGHT = 0.3


# ==================== INTERNAL HELPERS ====================

def _compute_radial_distances(
    card_centroids: np.ndarray,
    template_positions: np.ndarray,
    dealer_center: np.ndarray,
    angular_weight: float = ANGULAR_WEIGHT,
) -> np.ndarray:
    """
    Compute pairwise distances between card centroids and template positions
    using a radial metric: (1-w)*euclidean + w*(angle_diff * radius).

    For the dealer position (index 0) pure Euclidean distance is used.

    All inputs must be in the same coordinate space (normalized or pixel).

    Returns:
        distances: (n_cards, n_positions) array
    """
    n_cards = len(card_centroids)
    n_positions = len(template_positions)
    distances = np.zeros((n_cards, n_positions))

    for i, card_pos in enumerate(card_centroids):
        for j, template_pos in enumerate(template_positions):
            euclidean_dist = np.linalg.norm(card_pos - template_pos)

            # Dealer position (index 0): pure Euclidean
            if j == 0:
                distances[i, j] = euclidean_dist
                continue

            # Player positions: blend Euclidean with angular penalty
            card_angle = np.arctan2(
                card_pos[1] - dealer_center[1],
                card_pos[0] - dealer_center[0],
            )
            template_angle = np.arctan2(
                template_pos[1] - dealer_center[1],
                template_pos[0] - dealer_center[0],
            )

            angle_diff = abs(card_angle - template_angle)
            if angle_diff > np.pi:
                angle_diff = 2 * np.pi - angle_diff

            radius = np.linalg.norm(template_pos - dealer_center)
            angular_penalty = angle_diff * radius

            distances[i, j] = (
                (1 - angular_weight) * euclidean_dist
                + angular_weight * angular_penalty
            )

    return distances


# ==================== PUBLIC API ====================

def assign_cards_to_positions(
    detections: List[Dict],
    template_norm: np.ndarray = MANUAL_TEMPLATE_NORM,
    max_distance_norm: float = MAX_DISTANCE_NORM,
    angular_weight: float = ANGULAR_WEIGHT,
    image_w: int = IMAGE_W,
    image_h: int = IMAGE_H,
) -> Dict[int, list]:
    """
    Assign card detections to the nearest of 8 template seat positions.

    Args:
        detections:        List of detection dicts with 'polygon_center' in pixel coords.
        template_norm:     (8, 2) normalized template positions (0-1 per axis).
        max_distance_norm: Maximum normalized distance for a valid assignment.
        angular_weight:    Weight for the angular component of the distance metric.
        image_w, image_h:  Frame dimensions used to normalize detection coords.

    Returns:
        Dict mapping position_id (0=Dealer, 1-7=Players) -> list of detection dicts.
    """
    empty = {i: [] for i in range(N_POSITIONS)}
    if not detections:
        return empty

    # Normalize card centroids to [0, 1]
    raw_centers = np.array([det["polygon_center"] for det in detections], dtype=np.float64)
    card_centroids_norm = raw_centers / np.array([image_w, image_h], dtype=np.float64)

    dealer_center_norm = template_norm[0]

    distances = _compute_radial_distances(
        card_centroids_norm, template_norm, dealer_center_norm, angular_weight
    )

    assignments = np.argmin(distances, axis=1)
    min_distances = np.min(distances, axis=1)

    clusters: Dict[int, list] = {i: [] for i in range(N_POSITIONS)}
    for card_idx, (pos_idx, dist) in enumerate(zip(assignments, min_distances)):
        if dist <= max_distance_norm:
            clusters[pos_idx].append(detections[card_idx])

    return clusters


def get_active_players(detections: List[Dict]) -> str:
    """
    Cluster card detections and return the active-player string (e.g. "12357").

    Returns an empty string when there are no detections.
    """
    if not detections:
        return ""
    clusters = assign_cards_to_positions(detections)
    return "".join(str(p) for p in range(1, 8) if clusters[p])
