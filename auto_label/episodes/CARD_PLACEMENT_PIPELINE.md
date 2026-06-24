# Blackjack Card-Placement Pipeline

End-to-end documentation for the interactive card-placement annotation tool
(`web_app.py`) and the data / model pipeline that feeds it.

This document covers:

1. **What problem we are solving** and the overall data flow.
2. **How the raw data is collected** (videos → rounds → card detections).
3. **How the seat detector is trained** (YOLO Oriented Bounding Boxes).
4. **How per-round labels are produced** (`*_per_card.json`).
5. **How the web app works** (placement prediction, constraints, heatmaps, save).

---

## 1. The problem & overall flow

We are building **ground-truth card placements** for live-dealer blackjack tables.
On a crowded table, up to 7 players plus the dealer each receive cards, and as
the round proceeds players *hit* (take another card, extending their fan) or
*double* (one extra card at an angle). We want, for each round, a clean labeled
image showing *where each card sits* per seat.

Doing this entirely by hand is slow. The pipeline therefore **predicts** plausible
card placements from real detections and round statistics, and the web app lets a
human step the deal forward, sanity-check it, and save the result.

```
 raw video streams
        │  (cut into single-hand "rounds")
        ▼
 roundcut videos + per-frame CARD DETECTIONS  ──────►  <round>_card.jsonl
        │                                                   (rank/suit/box per card)
        │  YOLO-OBB seat detector  (trained separately)
        ▼
 seat OBBs (oriented polygons per seat, per round)
        │
        ▼  gen_intra_pile_labels.py / batch_intra_pile.py
 <round>_per_card.json   (tracks: which card → which seat, in what deal order)
        │
        ▼  web_app.py  (Flask UI)
 statistical initial placement + Hit/Double prediction + collision handling
        │
        ▼  Save
 card_placement_with_dealer/{json,image_original,overlay_image}/<round>.*
```

All coordinates are in a **1280×720** image space (videos are resized to this).

---

## 2. Data collection

### 2.1 Rounds ("roundcut")

A **round** is a single blackjack hand, cut out of a longer live-dealer video
stream. Each round is one short `.mp4` plus a name string like
`2025-10-01_06-05-15_000484_003464` (timestamp + frame span).

- Round videos live in
  `/home/ubuntu/us-west-3-fs/live_dealer_blackjack/roundcut/good_quality_rounds/<round>.mp4`.
- The web app reads the **first frame** of this video as the background canvas
  (`_first_frame`, cached under `/tmp/first_frames/`).

### 2.2 Card detections (`<round>_card.jsonl`)

An upstream card-detection model (trained separately from this pipeline; not
included here) runs on every frame of each round and writes a **line-delimited
JSON** file, one line per frame:

```jsonc
{"frame_id": 2594,
 "detections": [
   {"box": [578, 432, 642, 492],          // axis-aligned bbox [x1,y1,x2,y2]
    "polygon_center": [610, 462],          // card centroid (sometimes [[cx,cy]])
    "rank": "K", "suit": "S",              // classified rank + suit (letters)
    "conf": 0.91 },
   ...
 ]}
```

- Source dir: `/home/ubuntu/us-west-3-fs/live_dealer_blackjack/roundcut/cards_results/good_quality_rounds/all_jsons/<round>_card.jsonl`
  (this is `predict_next_card.DEFAULT_CARDS_DIR`).
- `rank` ∈ `{A,2..9,10,J,Q,K}`, `suit` ∈ `{S,H,D,C}`. **Card backs** are tagged
  `("CB","CB")` and are filtered out everywhere.
- **Rank/suit OCR is noisy**, so downstream tracking relies on *position*, not
  identity (see §4).

The `representative_frame` of a round (e.g. `round_fid=2594`) is the single frame
chosen to read seat geometry and the "settled" card positions from.

---

## 3. The seat detector (YOLO-OBB)

We need to know **which region of the table belongs to which seat**. Seats are
not axis-aligned (the table is viewed at an angle), so we use **Oriented Bounding
Boxes (OBB)** — rotated 4-corner polygons — with one class per seat:

```
Dealer=0, P1=1, P2=2, P3=3, P4=4, P5=5, P6=6, P7=7
```

### 3.1 How seat training data is created

The labels go through a *bootstrap-then-correct* loop:

1. **Auto pre-labels** (`gen_clustered_annotations.py`, `gen_to_annotate_prelabels.py`)
   - Input: extracted PNG frames (`…/clus_anno/to_annotate`) + the card-detection
     JSONL for each frame.
   - Cards in a frame are clustered to 8 template seat positions; each seat's
     cards are unioned into a bounding box (with a few px padding).
   - Output: a clustered-annotations JSON, and a **Label Studio** task file so a
     human can review/correct the boxes.

2. **Human correction (Label Studio)** produces rotated OBB corners per seat.

3. **Dataset assembly**
   - `build_balanced_dataset.py` — balances roughly equal numbers of
     human-corrected and auto-clustered examples (seeded split, seed `42`).
   - `build_expanded_dataset.py` — oversamples hard "split" frames
     (`split_mult=5`, `rest_mult=1`, `val_frac=0.15`), duplicating images as
     `_r1`, `_r2`, … for training while validation uses originals only. Adds an
     explicit `"split": "train"|"val"` field.

### 3.2 Training (`train_yolo_obb.py`)

- Converts the annotations JSON into the **Ultralytics YOLO-OBB** layout, mirroring
  the official DOTA → YOLO-OBB pipeline:
  ```
  <out>/
    images/{train,val}/<png>                     # symlinked frames
    labels/{train,val}_original/<txt>            # DOTA: 8 abs coords + class + diff
    labels/{train,val}/<txt>                     # YOLO-OBB: class_idx + 8 normalized coords
    dataset.yaml
  ```
- Split: honors an explicit `"split"` field if present, else random 85/15 (seed `42`).
- Model & params: base weights `yolo26n-obb.pt`, `epochs=100`, `imgsz=1280`,
  `batch=8`, `device=0`.
- Output weights: `…/sahithi/yolo_obb_runs/<name>/weights/best.pt`.
  The downstream label generator uses the run named **`human_v2`**.

### 3.3 Inference helpers (debug/visualization only)

- `infer_yolo_obb.py` — run the detector on a folder of PNGs, save annotated PNGs.
- `infer_obb_on_videos.py` — annotate whole videos with seat polygons (`conf=0.3`).
- `infer_obb_on_clips.py` — same, restricted to `start..end` frame ranges.

These are not part of the per-round label path; they exist to eyeball detector
quality.

---

## 4. Per-round labels (`<round>_per_card.json`)

This is the file the web app actually loads. It is produced by
`gen_intra_pile_labels.py` (single round) / `batch_intra_pile.py` (many rounds),
using helpers in `intra_pile_util.py`.

### 4.1 How it's built

1. Probe the round's top-K frames by detection count; run the **YOLO-OBB seat
   model** (`human_v2/weights/best.pt`) until ≥8 seats are found → gives
   `seat_obbs` (4 corners per seat) and the `representative_frame`.
2. **`track_cards`** — greedy nearest-neighbour tracking of card centroids across
   frames (position-based, since rank/suit OCR is unreliable). Spurious blips are
   dropped by minimum duration / observation count.
3. **`assign_seats`** — for each track, majority vote of `point_in_obb` across its
   observations against every seat polygon; if it never lands inside, fall back to
   the nearest OBB within ~30 px.
4. **`assign_ordinals`** — within a seat, sort tracks by when they first appear
   *inside* the seat OBB and number them `1, 2, 3, …`. **Ordinal = deal order**, so
   ordinal 1 and 2 are the player's *initial two cards*; 3+ are hits/doubles.
5. **`assign_doubles`** — flags perpendicular "double" cards.

### 4.2 Schema (grounded in a real file)

```jsonc
{
  "round": "2025-10-01_06-05-15_000484_003464",
  "representative_frame": "round_fid=2594",
  "image_size": null,                       // null in practice; app assumes 1280x720
  "seat_obbs": {                            // only seats actually present appear
    "0": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], // 4 corners, seat 0 = Dealer
    "2": [...], "3": [...], ...
  },
  "tracks": [
    {
      "track_id": 0,
      "rank": "2", "suit": "S",             // majority-vote identity (letters)
      "seat": 0, "seat_name": "Dealer",
      "seat_via": "inside",                 // or "nearest(<px>)"
      "ordinal": 1,                         // deal order within the seat
      "double": false,
      "first_frame": 456, "last_frame": 2838,
      "first_center": [644.2, 442.4],
      "last_center":  [356.9, 502.3],       // final observed centroid (px)
      "n_observations": 2205,
      "mean_conf": 0.910
    }
  ]
}
```

Notes:
- `seat_obbs` keys are **strings**; the dealer is `"0"`.
- A seat only appears if it had ordinal-tagged tracks in that round — this is what
  the app calls **"seats present"**.
- `last_center` is the per-card position the heatmap/statistics are built from.

### 4.3 Related analysis scripts (not required by the app)

- `compute_meta_text_roundcut.py` — annotates timeline segments with the active
  player list.
- `find_adjustment_rounds.py` → `render_all_adjustments.py` /
  `viz_initial_hands_adjustment.py` — detect and visualize rounds where a card was
  physically *adjusted* (its stable position differs from its final position by
  ≥8 px). Useful for finding tricky rounds.
- `predict_next_card.py` — the original standalone "where does the next card land"
  predictor. The web app reuses its constants (`DOUBLE_ANGLE_DEG`,
  `HIT_ANGLE_DEG`, `STEP_MAGNITUDE_PX`) and `_rotate`.

---

## 5. The web app (`web_app.py`)

A Flask app on port **5050**. It loads a round's `*_per_card.json`, lays out the
initial deal, and lets a human extend each seat with Hit/Double, then save.

Run:
```bash
python web_app.py        # then open http://<host>:5050/
```

### 5.1 Key inputs

| Constant | Path | Purpose |
|---|---|---|
| `OUT_DIR` | `…/clus_anno/intra_pile_v1_double_only_1k` | the `*_per_card.json` rounds |
| `DEFAULT_CARDS_DIR` | `…/roundcut/.../all_jsons` | `<round>_card.jsonl` detections |
| `POLYDRAW_PATH` | `…/clus_anno/polydraw_json/polydraw-merged.json` | seat **card-shape polygons** |
| `VIDEO_DIR` | `…/roundcut/good_quality_rounds` | round videos (first frame = canvas) |
| `SPLIT_BLOCKLIST_PATH` | `…/intra_pile_v1_split_rounds.txt` | rounds to hide from the list |

The **polydraw** file defines the *shape of one card* per seat (a small rotated
rectangle, stored as corner offsets around a center). It is editable in-browser at
`/annotate` (upload a frame, click 4 corners per seat, Save). Every placed card is
this polygon translated to a center point.

### 5.2 Statistical initial placement

Rather than trust a single round's noisy detections, the initial 2 cards per seat
are placed at the **population median** position for that `(seat, ordinal-index)`:

- `_statistical_initial_centers()` scans **all listed rounds**, takes each seat's
  ord-1/ord-2 `last_center` (`_raw_initial_centers_fast`), keeps only points that
  fall **inside that seat's OBB** (rejects bad matches), and takes the median x and
  y per `(seat, idx)`.
- The result is cached in memory **and** on disk at `/tmp/stat_initial_cache.json`,
  so restarts are instant. A lock prevents concurrent rebuilds.
- **Dealer (seat 0)** is special: its two cards are pinned side-by-side around the
  dealer's median position, spaced by the dealer card width so they just touch
  (`_dealer_initial_centers`, `_dealer_step_px`).

`_initial_centers()` returns these medians; the round only *shows* them after the
user clicks **Initial hand** (`placed_initial[round] = True`).

### 5.3 "Initial hand" animation

The **Initial hand ▶** button animates the deal in true table order: pass 1 deals
card 1 to P1…P7 then Dealer, pass 2 deals card 2 in the same order — skipping seats
not present in this round (`_deal_order_for`, `_seats_present`). Each step renders a
PNG (`/initial_hand/<round>/<step>.png`, cached under `/tmp/initial_hand_cache/`),
and the client flips through them ~350 ms apart.

### 5.4 Hit / Double prediction

When the user clicks **Hit** or **Double** for a seat (`_predict_for_seat`):

- **Players:** the fan direction `v0` is the vector from initial card 1 → card 2,
  normalized to `STEP_MAGNITUDE_PX`. A **Hit** steps along `v0` (angle 0); a
  **Double** steps at `DOUBLE_ANGLE_DEG` from `v0`. The new card is appended to the
  seat's `extras` list with its polygon.
- **Dealer:** cards just continue side-by-side by the dealer card width (no angle).
- Rules enforced: a seat needs its initial 2 cards first; **Double** requires
  exactly 2 cards and is disallowed for the dealer.

### 5.5 Collision handling & constraints

After every placement, `_rebalance_all` (up to `REBALANCE_MAX_PASSES=5`) keeps the
layout physically plausible. Per seat (`_rebalance_seat`):

- **Horizontal displacement only.** If a seat's cards overlap another seat's
  (bbox overlap > `OVERLAP_RATIO_THRESHOLD=0.1`), the *whole cluster* is shifted
  left/right (`CLUSTER_DISPLACE_STEP_PX=4`, capped at `CLUSTER_DISPLACE_MAX_PX=80`)
  away from the offender. The displacement is **persisted** in `seat_displacements`
  so it doesn't snap back, and it also moves the initial 2 cards so the cluster
  moves as one unit. Vertical displacement is disallowed.
- **Cluster shrink** (`_cluster_shrink_ratio`): the hit-chain is contracted
  *uniformly* (every card's offset from card 2 scaled by one ratio, never below
  `MIN_SHRINK_RATIO=0.15`) so that:
  - the chain stays within `MAX_CLUSTER_DY_PX=120` of card 2 in y, and
  - no card rises above `MIN_ABS_Y_PX=470` (cards stay in the players' band).
- **Dealer clearance** (`_dealer_clear_ratio`): player chains additionally shrink
  so no player card bbox overlaps any dealer card bbox (with `DEALER_CLEARANCE_PX=4`
  margin). The dealer is never displaced or shrunk — it is the fixed anchor.

### 5.6 Heatmap

The **Initial-card heatmap** toggle overlays where initial cards land across *all*
rounds: red (`CARD1_COLOR_BGR`) for card 1, cyan (`CARD2_COLOR_BGR`) for card 2.
Densities are accumulated per pixel (filtered to inside-OBB points), Gaussian-
blurred, normalized, and blended over the first frame. Cached in memory and on disk
(`/tmp/heatmap_cache/`). Optional `?seat=<n>` filter restricts to one seat.

### 5.7 Saving

The green **Save** button (enabled only after Initial hand) calls `/save/<round>`
and writes three artifacts per round under
`…/clus_anno/card_placement_with_dealer/`:

| Artifact | Path | Contents |
|---|---|---|
| Labels | `json/<round>.json` | per seat → `cards[]` with `ordinal`, `kind` (`initial`/`hit`/`double`), `center`, `polygon` (4 corners), `bbox` |
| Clean frame | `image_original/<round>.png` | the first video frame, unannotated |
| Overlay | `overlay_image/<round>.png` | first frame with all predicted polygons drawn |

The saved JSON merges the (possibly displaced) statistical initial cards with the
user's hit/double extras, so it captures exactly what is on screen.

### 5.8 HTTP routes

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | main UI (round picker, seat table, heatmap toggle) |
| `/round/<name>` | GET | seat states (card counts, can_hit/can_double, present seats) |
| `/predict/<name>` | POST | apply a Hit/Double for a seat, returns new state |
| `/place_initial/<name>` | POST | mark the initial hand as placed |
| `/reset/<name>` | POST | clear session extras + displacements |
| `/save/<name>` | POST | write the 3 artifacts |
| `/vis/<name>.png` | GET | current rendered overlay |
| `/initial_hand/<name>/<step>.png` | GET | animation frame |
| `/heatmap.png` | GET | initial-card heatmap (optional `?seat=`) |
| `/annotate`, `/annotate/{upload,img,save}` | GET/POST | edit seat card polygons |

### 5.9 In-memory state

All session state is per-process (no DB):
- `sessions[round] = {seat_id: [{center, polygon, is_double}, …]}` — placed extras.
- `placed_initial[round]` — whether the initial deal is shown.
- `seat_displacements[(round, seat)]` — persisted cluster shift.
- Various caches: `_round_cache`, `_rep_frame_cache`, `_bbox_cache`,
  `_list_rounds_cache`, `_heatmap_cache`, plus the on-disk caches in `/tmp/`.

`/reset` clears the session, placed flag, and displacements for a round.

---

## 6. Quick start

```bash
cd /home/ubuntu/sahithi/motion-data-process/auto_label/episodes
python web_app.py
# open http://<host>:5050/
# 1. pick a round   2. (optional) toggle heatmap   3. Initial hand ▶
# 4. Hit/Double per seat as needed   5. Save
```

To (re)generate per-round labels for new rounds:
```bash
python batch_intra_pile.py --rounds_file <rounds.txt>   # writes *_per_card.json
```

To retrain the seat detector after correcting annotations:
```bash
python train_yolo_obb.py --ann_json <annotations.json> --name <run_name>
# best weights -> .../yolo_obb_runs/<run_name>/weights/best.pt
```
