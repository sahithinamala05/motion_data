# Hit Segment Processing

Processes `instance_segments.json` from FACT_actseg to extract `hit` / `dealer hits` segments with associated card detection bounding boxes, matching the manual label format.

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
Output annotations JSON (per video)
```

Each stage is independent; failures at every stage are recorded and visualised separately.

## Usage

```bash
# Full run (filtering + failure vis, no success vis)
python process_hit_segments.py

# Full run + visualise 20 random successful segments
python process_hit_segments.py --vis_count 20

# Full run + visualise all successful segments
python process_hit_segments.py --vis_count -1

# Standalone: re-run success vis from annotation JSONs (no re-filtering)
python process_hit_segments.py --vis_only --vis_count 20

# Custom output dir for success vis clips
python process_hit_segments.py --vis_count 20 --vis_out_dir /path/to/vis_out
```

### CLI Options

| Option | Default | Description |
|---|---|---|
| `--vis_count` | `0` | Success clips to render: `0`=none, `N`=random N, `-1`=all |
| `--vis_out_dir` | `./vis_out` | Directory for success visualisation videos |
| `--vis_only` | off | Skip filtering; load `vis_segments_cache.json` and render vis only |

## Output

All outputs written to `output_hit_segments_2k5/`.

```
output_hit_segments_2k5/
├── <video>_annotations.json       # one per video; hit/dealer-hits segments >= 30 frames
│                                  # bounding_boxes populated only for stage-3 passing segments
├── stats.json                     # success/failure counts by stage
├── failure_cases.json             # all failure records
├── failure_vis/
│   ├── stage1_duration/           # raw clips of short segments
│   ├── stage2_no_new_card/        # clips with start(blue)/end(green) det overlays
│   ├── stage2_no_detections/
│   ├── stage2_bad_frame_order/
│   └── stage3_spatial/            # clips with rejected cards(red) + constraint region(orange)
│
└── (vis_out/ or --vis_out_dir)    # success vis clips (if --vis_count != 0)
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
