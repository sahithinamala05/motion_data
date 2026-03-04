# 10s-chunk Auto-Label Pipeline

```
compute_meta_text_and_export_v2.py   ← uses clustering_util.py
        ↓
form_action_data_roundwise.py
```

---

## `clustering_util.py`
Shared clustering utilities (no CLI). Imported by other scripts.

- **`get_active_players(detections)`** — returns active-player string (e.g. `"1357"`)
- **`assign_cards_to_positions(detections)`** — assigns detections to nearest seat using radial distance (Euclidean + angular penalty around dealer)
- **`MANUAL_TEMPLATE_NORM`** — 8 seat centroids (Dealer + P1–P7) normalized to `[0,1]` from 1920×1080 pixel coords:

```
Dealer: [972, 723]   P1(R): [1389, 777]   P2: [1257, 829]   P3: [1125, 861]
P4: [963, 892]       P5: [796, 879]        P6: [688, 837]    P7(L): [562, 778]
```

---

## `compute_meta_text_and_export_v2.py`
For each prediction JSON, finds the `"initial hands 1st"` segment, looks up YOLO card detections at its last frame, clusters them → active player seats, and writes `meta_text` (e.g. `"1357"`) into `"initial hands 1st"`, `"initial hands 2nd"`, and `"discard"` segments. Saves updated JSONs to `predictions_json_with_meta/`.

Key config (top of `main()`):

| Variable | Description |
|---|---|
| `PRED_JSON_DIR` | Input prediction JSONs |
| `CARD_DETECTION_DIR` | YOLO card detection JSONs |
| `OUTPUT_DIR` | Output directory for updated JSONs |

```bash
python compute_meta_text_and_export_v2.py
```

---

## `form_action_data_roundwise.py`
For each prediction JSON, merges the overlapping per-300-frame dwpose PKL files into a single per-round PKL trimmed to the clip's frame boundaries. Supports resume (skips already-written output files). Set `ANNOTATION_DIR = ""` to skip annotations and process predictions only.

Key config (top of `main()`):

| Variable | Description |
|---|---|
| `PKL_DIR` | Source per-300-frame dwpose PKLs |
| `ANNOTATION_DIR` | Annotation JSONs (set `""` to skip) |
| `PREDICTION_DIR` | Prediction JSONs (set `""` to skip) |
| `OUTPUT_BASE` | Root output directory |

```bash
# Full run (resumes automatically if interrupted)
python form_action_data_roundwise.py

# Process + visualize a few clips first
python form_action_data_roundwise.py --examples-only 3

# Full run + visualize N random examples after
python form_action_data_roundwise.py --vis-examples 5
```

---

## File Naming Convention

```
YYYY-MM-DD_HH-MM-SS_XXXXXX_YYYYYY[_suffix].json/.pkl
                     ^^^^^^  ^^^^^^
                     clip index   starting frame (1-indexed)
```

Each clip covers 300 frames: `YYYYYY` to `YYYYYY + 299`.
