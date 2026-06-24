# `initial_hand/` — initial-hand card-layout predictions

Per-round JSON: for each **"initial hands"** segment of a blackjack round, which
cards are on the table, at which seat, and in what dealing order — tracked per frame
and run-length-compressed.

- **Location:** `/home/ubuntu/us-west-3-fs/sahithi/initial_hand/`
- **Files:** `18,914` × `<round>_predictions.json` (full ~26k autolabel batch).
- **Human-adjusted subset:** 378 files, same schema, in `initial_hand_adj/` .

---

## 1. JSON schema

```jsonc
{
  "video_name": "2025-10-01_06-05-15_003465_006034.mp4",
  "clip_start_1idx": 3465,            // round's 1st frame in the recording (1-based)
  "initial_hands": [                  // 1–2 entries: "initial hands 1st" / "2nd"
    {
      "label": "initial hands 1st",
      "start_frame": 815, "end_frame": 989,        // LOCAL (relative to clip)
      "global_start_frame": 4279, "global_end_frame": 4453,  // global = clip_start_1idx-1 + local
      "meta_text": ["123456"],        // active player seats, digits 1–7 (dealer excluded)
      "card_layout": [                // run-length-encoded card config; n_cards rises 0→1→2…
        {
          "start_frame": 4279, "end_frame": 4299,  // GLOBAL frames
          "n_frames": 21,
          "n_cards": 1,
          "cards": [                  // representative cards at the run's settled frame
            {
              "centroid": [1104.7, 661.64],  // card center [x,y], 1920×1080 px
              "role": "dealer",              // "dealer" (seat 0) | "player" (1–7)
              "seat": 0,                     // 0=dealer, 1=rightmost player … 7=leftmost
              "confidence": 0.889,           // YOLO confidence
              "order": 6                     // dealing order, 0-based, stable per card
            }
          ]
        }
      ]
    }
  ]
}
```

**Frames:** segment `start_frame`/`end_frame` are **local**; `global_*` and all
`card_layout` frames are **global**. `global = (clip_start_1idx - 1) + local`.

**`card_layout` = runs**, not per-frame: each run is a span of frames with the same
seat occupancy (blips < 3 frames debounced); its `cards` come from the run's last
settled frame.

**Fields:** `seat` 0=dealer (top-center), 1=rightmost player … 7=leftmost. `order` =
blackjack dealing order, fixed per physical card (players seat 1→7 then dealer, per
round); dealer is never `order` 0. Cards below confidence 0.8 are dropped.

---

## 2. Read it

```python
import json, glob
d = json.load(open(glob.glob("/home/ubuntu/us-west-3-fs/sahithi/initial_hand/*_predictions.json")[0]))
for hand in d["initial_hands"]:
    final = max(hand["card_layout"], key=lambda r: r["n_cards"])   # fully-dealt layout
    print(hand["label"], "seats", hand["meta_text"])
    for c in sorted(final["cards"], key=lambda c: c["order"]):
        print(f'  order {c["order"]:>2}  seat {c["seat"]} ({c["role"]})  conf {c["confidence"]}')
```

---

## 3. How it was generated

**Generator:** `initial_hand/code/compute_initial_hand_cards.py`. Shared helpers
(`clustering_util.py`, `compute_meta_text_and_export_v2.py`) stay in
`auto_label/10s-chunk/` (other scripts import them); the generator adds that dir to
`sys.path`.

**Inputs:** autolabel `timeline_segments` from `AutoLabeling_batch_01_part_1/`
(initial-hand segments + active-seat `meta_text`) + per-frame YOLO card detections
from `live_dealer_blackjack/yolo_cards/card_detection/all_jsons/`.

Per round it keeps the `initial hands 1st/2nd` segments, looks up + confidence-filters
(`≥0.8`) card detections each frame, clusters them to the 8 seats, run-length-encodes
into `card_layout` (3-frame debounce), and assigns a stable dealing `order`.

**Re-run:**
```bash
cd /home/ubuntu/sahithi/motion-data-process/initial_hand/code
python compute_initial_hand_cards.py      # paths are constants at top of file; needs tqdm
```

---

## 4. Related directories

| path | what it is |
|---|---|
| `/home/ubuntu/us-west-3-fs/sahithi/initial_hand/` | this — autolabel predictions, full batch (18,914 files) |
| `/home/ubuntu/us-west-3-fs/sahithi/initial_hand_adj/` | human-adjusted subset, 378 files, same schema. Use as ground-truth-quality labels. |
| `/home/ubuntu/us-west-3-fs/sahithi/initial_hand_adj_player_index/` | same 378 rounds in upstream `timeline_segments` format. Built via `/home/ubuntu/sahithi/motion-data-process/initial_hand/code/convert_initial_hand_adj_to_predictions.py` from the Label Studio export. |
| `/home/ubuntu/us-west-3-fs/sahithi/initial_hand_vis/` · `/home/ubuntu/us-west-3-fs/sahithi/initial_hand_adj_vis/` | 50 overlay `.mp4` clips each, visualizing the runs on the source video. |
| `/home/ubuntu/us-west-3-fs/live_dealer_blackjack/action_annotation/mixed_mini_batch/annotation/AutoLabeling_batch_01_part_1/` | upstream machine input (`timeline_segments` + `meta_text`, 19,917 files). |
