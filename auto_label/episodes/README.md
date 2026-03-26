# Episode Segment Processing

Scripts for extracting action segments from FACT_actseg `instance_segments.json` with associated card detection bounding boxes, matching the manual label format.

---

## Hit / Dealer Hits (`process_hit_segments.py`)

Extracts `hit` / `dealer hits` segments with bounding boxes via new-card detection.

Supports **resuming after interruption**: already-written annotation JSONs are skipped on re-run.

## Filtering Pipeline

```
all segments
    │
    ▼
Stage 1 – duration
    Keep hit/dealer-hits with >= 30 frames.
    │
    ▼
Stage 2 – new card diff
    Load card detection JSONL for the video.
    Compare start-frame vs end-frame detections via greedy nearest-neighbour
    matching (threshold 120 px). End-frame cards with no match = newly appeared.
    Fail if no new card found.
    │
    ▼
Stage 3 – spatial
    dealer hits : centroid must be inside pixel bbox (535,411)–(800,489);
                  if multiple, pick highest-x card.
    hit         : centroid y >= 450 px; if multiple, pick highest-y card.
    │
    ▼
Output annotations JSON (per video, written immediately for resume support)
```

Each stage is independent; failures at every stage are recorded and optionally visualised.

## Usage

```bash
# Full run (filtering only, no vis)
python process_hit_segments.py

# Full run + visualise 20 random successful segments
python process_hit_segments.py --vis_count 20

# Full run + visualise all successful + 10 random failure segments
python process_hit_segments.py --vis_count -1 --fail_vis_count 10

# Resume an interrupted run (already-written JSONs are skipped automatically)
python process_hit_segments.py

# Standalone: re-run success vis from annotation JSONs (no re-filtering)
python process_hit_segments.py --vis_only --vis_count 20
```

### CLI Options

| Option | Default | Description |
|---|---|---|
| `--vis_count` | `0` | Success clips to render: `0`=none, `N`=random N, `-1`=all |
| `--fail_vis_count` | `0` | Failure clips to render: `0`=none, `N`=random N, `-1`=all |
| `--vis_only` | off | Skip filtering; reconstruct vis segments from annotation JSONs and render success vis only |

## Output

All outputs written to `output_hit_segments_2k5/`.

```
output_hit_segments_2k5/
├── <video>_annotations.json       # one per video; hit/dealer-hits segments >= 30 frames
│                                  # bounding_boxes populated only for stage-3 passing segments
├── stats.json                     # success/failure counts by stage
├── failure_cases.json             # all failure records
├── failure_vis/                   # (if --fail_vis_count != 0)
│   ├── stage1_duration/           # raw clips of short segments
│   ├── stage2_no_new_card/        # clips with start(blue)/end(green) det overlays
│   ├── stage2_no_detections/
│   ├── stage2_bad_frame_order/
│   └── stage3_spatial/            # clips with rejected cards(red) + constraint region(orange)
│
└── success_vis/                   # (if --vis_count != 0)
```

### Annotation JSON format

```json
{
  "video_name": "video.mp4",
  "video_id": "",
  "video_path": "",
  "timeline_segments": [
    {
      "start_frame": 100,
      "end_frame": 145,
      "labels": ["hit"],
      "meta_text": [],
      "duration_frames": 46,
      "id": "<random 10-char>",
      "bounding_boxes": [
        {
          "frame": 143,
          "labels": ["card"],
          "keyframes": [{
            "frame": 143,
            "x": 45.2, "y": 60.1, "width": 5.0, "height": 8.3,
            "polygon_center": [580, 433],
            "box": [578, 432, 642, 492],
            "conf": 0.91,
            "rank": "K", "suit": "S"
          }],
          "id": "<random 10-char>"
        }
      ]
    }
  ]
}
```

`x/y/width/height` are percentages of 1280×720. `bounding_boxes` is `[]` for segments that failed stage 2 or 3.

---

## Call for Action (`process_call_for_action_segments.py`)

Extracts `call for action` segments with bounding boxes determined by index-finger-tip proximity to card centroids using dwpose hand tracking.

Supports **resuming after interruption**: already-written annotation JSONs are skipped on re-run.

### Filtering Pipeline

```
all segments
    │
    ▼
Stage 1 – duration
    Keep call-for-action with >= 45 frames.
    │
    ▼
Stage 2 – pose sampling
    Load dwpose pkl; sample every 5 frames within segment.
    Extract index finger tip (keypoint 8) for both hands.
    Look up card detections at each sampled frame (±2 frame fallback).
    Only consider cards with centroid y > 450 px.
    Fail if no frame has both hands detected + eligible card detections.
    │
    ▼
Stage 3 – card vote (two-criterion selection)
    C1  = card nearest to a finger tip most often across sampled frames (vote winner).
    C2  = card with globally smallest distance to any tip in any frame.
    C_final = C2 if Dmin_global < Dmin_C1 and C1 ≠ C2 (>20 px apart);
              C1 otherwise.
    │
    ▼
Output annotations JSON (per video, written immediately for resume support)
```

### Usage

```bash
# Full run (filtering only, no vis)
python process_call_for_action_segments.py

# Full run + visualise 50 random successful segments
python process_call_for_action_segments.py --vis_count 50

# Full run + 20 success vis + 10 failure vis
python process_call_for_action_segments.py --vis_count 20 --fail_vis_count 10

# Resume an interrupted run (already-written JSONs are skipped automatically)
python process_call_for_action_segments.py
```

### CLI Options

| Option | Default | Description |
|---|---|---|
| `--vis_count` | `0` | Success clips to render: `0`=none, `N`=random N, `-1`=all |
| `--fail_vis_count` | `0` | Failure clips to render: `0`=none, `N`=random N, `-1`=all |

### Output

All outputs written to `output_cfa_segments_2k5/`.

```
output_cfa_segments_2k5/
├── <video>_annotations.json       # one per video; CFA segments >= 45 frames
│                                  # bounding_boxes populated only for stage-3 passing segments
├── stats.json                     # success/failure counts by stage
├── failure_cases.json             # all failure records
├── failure_vis/                   # (if --fail_vis_count != 0)
│   ├── stage1_duration/
│   ├── stage2_no_pkl/
│   ├── stage2_no_card_jsonl/
│   ├── stage2_no_valid_samples/
│   └── stage3_no_votes/
│
└── success_vis/                   # (if --vis_count != 0)
    # clips with: chosen card (green bbox), index tips (cyan=left, magenta=right),
    # lines from tips to chosen centroid, all other cards (blue), vote info overlay
```

### Visualisation overlay

- **Green bbox + centroid**: chosen card (C_final)
- **Cyan dot**: left hand index finger tip (on sampled frames)
- **Magenta dot**: right hand index finger tip (on sampled frames)
- **Blue bboxes**: all other detected cards at that frame
- **Top-left text**: selection method (`vote_winner` / `global_closest`), vote ratio, Dmin_C1, Dmin_global
