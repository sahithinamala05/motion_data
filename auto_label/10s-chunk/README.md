# 10s-chunk Auto-Label Pipeline

Scripts for processing 10-second video clips: downloading, filtering, computing
active-player metadata via card clustering, and exporting training data.

---

## Pipeline Overview

```
download_filtered_videos.py
        ↓
find_bad_data_and_filter_videos.py
        ↓
compute_meta_text_and_export_v2.py   ← uses clustering_util.py
        ↓
form_action_data_roundwise.py  →  form_final_action_data.py
```

---

## Scripts

### `download_filtered_videos.py`
Downloads blackjack video clips from S3, filtering by minimum duration.

Key config (top of file):
| Variable | Default | Description |
|---|---|---|
| `MIN_DURATION_SECONDS` | 30 | Skip clips shorter than this |
| `MAX_VIDEOS_TO_DOWNLOAD` | None | Cap on successful downloads (None = all) |
| `RANDOM_SELECTION` | True | Random selection when cap is set |
| `RESUME_DOWNLOADS` | True | Skip already-downloaded files |

```bash
python download_filtered_videos.py
```

---

### `find_bad_data_and_filter_videos.py`
Merges per-300-frame dwpose PKL files into per-round PKL files and flags bad clips.

```bash
python find_bad_data_and_filter_videos.py
```

---

### `compute_meta_text_and_export_v2.py`
**Main labeling script.** For each prediction JSON:
1. Finds the `"initial hands 1st"` segment
2. Looks up YOLO card detections at that segment's last frame
3. Clusters detections → active player seats (e.g. `"1357"`)
4. Writes `meta_text` into `"initial hands 1st"`, `"initial hands 2nd"`, and `"discard"` segments
5. Saves updated JSONs to `predictions_json_with_meta/`

Clustering logic is imported from `clustering_util.py`.

```bash
python compute_meta_text_and_export_v2.py
```

Key config:
| Variable | Description |
|---|---|
| `PRED_JSON_DIR` | Input prediction JSONs |
| `CARD_DETECTION_DIR` | YOLO card detection JSONs |
| `OUTPUT_DIR` | Output directory for updated JSONs |

---

### `clustering_util.py`
Shared clustering utilities (no CLI).  Import in other scripts.

- **`MANUAL_TEMPLATE_NORM`** — 8 seat centroids (Dealer + P1–P7), normalized to `[0,1]`
  from original 1920×1080 pixel coords.
- **`assign_cards_to_positions(detections)`** — assigns each detection to the nearest seat
  using a radial distance metric (Euclidean + angular penalty around the dealer).
- **`get_active_players(detections)`** — returns active-player string (e.g. `"1357"`).

Seat layout (1920×1080 pixel coords):

```
Dealer: [972, 723]   P1(R): [1389, 777]   P2: [1257, 829]   P3: [1125, 861]
P4: [963, 892]       P5: [796, 879]        P6: [688, 837]    P7(L): [562, 778]
```

---

### `visualize_clustering.py`
Visualizes the clustering result at the exact frame used for `meta_text` computation.
Saves annotated PNGs (seat circles, card X-marks, assignment lines, active-player label).

```bash
# 5 random files (default)
python visualize_clustering.py

# Specific number, random
python visualize_clustering.py --n 20

# Specific number, alphabetical order
python visualize_clustering.py --n 10 --no-random

# One specific file
python visualize_clustering.py -f 2024-01-15_12-30-00_001234_001534_predictions.json

# Named file + 4 more random (5 total)
python visualize_clustering.py -f my_file.json --n 5

# Multiple named files
python visualize_clustering.py -f file_a.json -f file_b.json
```

Output PNGs are saved to `data/.../clustering_vis/` by default.
Override with `--output-dir`, `--pred-dir`, `--card-dir`.

---

### `form_action_data_roundwise.py`
Merges per-300-frame dwpose PKL files into per-round PKL files aligned to clip boundaries.

```bash
# Process a few clips and visualize
python form_action_data_roundwise.py --examples-only 3

# Full run
python form_action_data_roundwise.py

# Full run + visualize N random examples
python form_action_data_roundwise.py --vis-examples 5
```

---

### `form_final_action_data.py`
Combines dwpose NPY files with annotation JSONs to produce per-segment PKL training data.

```bash
python form_final_action_data.py
```

---

## File Naming Convention

```
YYYY-MM-DD_HH-MM-SS_XXXXXX_YYYYYY[_suffix].json/.mp4/.npy/.pkl
                     ^^^^^^  ^^^^^^
                     clip index (0-indexed)
                     starting frame (1-indexed in S3 round_cut files)
```

Each clip covers 300 frames: frames `YYYYYY` to `YYYYYY + 299`.
