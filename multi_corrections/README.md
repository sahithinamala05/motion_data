# multi_corrections

Isolate the **front-table dealer** from multi-person WiLoR/DWPose blackjack clips, and QA/quality-filter them.
Env: `/home/ubuntu/miniconda3/envs/wilor/bin/python`. Inputs: DWPose+WiLoR pkls in
`Wilor/video_cut_wilor_rem/`, videos in `live_dealer_blackjack/video_cut/batch_01/`,
DB `db/db_batch_01.parquet`. Outputs under `multi_correction/full_run/` and `multi_correction/videos/`.

## Pipeline (top level)

- **`dealer_faceid.py`** — MAIN. Keeps only the middle/front-table dealer's DWPose body + ≤2 hands + face
  per frame; same schema as input (drop-in). Reference-free, no fallback, **pure DWPose/WiLoR (no insightface/GPU)**.
  - Step 1 front-table = big FACE (`face_size` ≥ 0.6 × clip 80th-pct face size); far tables ignored.
  - Step 2 anchor = median central front-table head; per frame keep the body at the anchor, gated by
    ≥66/68 DWPose face keypoints AND ≥8 body joints; hands = per-wrist nearest (≤2), drop if closer to a
    behind person's arm.
  - Run: `python dealer_faceid.py --stem <s> [--viz]` or batch `--shard i --nshards N --viz_first 100`.
  - Full run: `launchers/run_final.sh` (16 CPU shards, ~10 min) -> `full_run/pkl/`, `full_run/vis/`.

- **`filter_quality.py`** — dark + blur quality filter for the multi clips (improves the stage-2 face filter):
  explicit DARK frame removal (mean intensity < 30) + PROPORTION-based blur (`cv2.Laplacian`.var ≥ 100 over
  evenly-sampled frames, not early-exit at 30). Emits per-clip `frac_dark/frac_blurry/frac_good/pass`, joined
  with the DB's `ocr_passed/face_passed` -> `videos/quality_multi.parquet`. Run: `launchers/run_quality.sh`.

- **`filter_quality_perframe.py`** — per-FRAME version of the above. Analyses EVERY frame and stores each clip's
  `problematic_segments` (dark OR blurry) plus `dark_segments`/`blurry_segments` as `[start_f, end_f, start_s, end_s]`
  intervals (@30fps), and `frac_dark/blurry/good`. **Nothing is removed** — clips are kept with their problematic
  ranges so downstream can decide per action-window (a 10s chunk must be clean *throughout the annotated action*).
  `pass` (frac_good>=0.5) is informational only. **OCR is NOT handled here** — it's Yifan/Dassie's separate
  project-specific phrase filter (`LiveDealer_data_processing/ocr_script`), joined later; we don't gate on DB `ocr_passed`.
  Scope = **ALL 94,004** remaining clips = 189K minus project21 `video_cut` (95K). **No face-based pre-filter** —
  `face_passed` bakes in the over-strict face-similarity check that wrongly rejects good occluded clips, so we keep
  everything and let per-frame dark/blur (+ OCR later) decide (recover-everything approach).
  Run: `launchers/run_perframe.sh` (24 shards) -> `rem_95k_filtered/{quality_perframe.parquet, quality_summary.txt}`.

- **`apply_quality_to_pkl.py`** — applies the finalised per-frame quality result to the single-dealer pkls:
  for each `full_run/pkl/<cid>_dealer.pkl`, nulls the pose on the dark/blur frames (from
  `rem_95k_filtered/quality_perframe.parquet` `problematic_segments`) -> `full_run/pkl_updated/<cid>_dealer.pkl`.
  A problematic frame becomes `{'pose': None, 'removed_dark_blur': True, 'frame_dimensions': ...}`; good frames
  unchanged. FAST (pkl edits, no decode); frame indices align 1:1 (pkl frame i == video frame i == quality index).
  **Run only AFTER `quality_perframe.parquet` is finalised.** Sharded + resumable: `launchers/run_apply_pkl.sh`
  (16 shards) or `python apply_quality_to_pkl.py --shard i --nshards N`. Comparison viz (left OLD | right NEW,
  full_run/vis style, 3840x1080 H.264) for a date-spread sample: `python apply_quality_to_pkl.py --viz --nviz 20`
  -> `full_run/pkl_updated/vis/`.

- `dealer_match.py`, `multi_ref_frames.py` — earlier/auxiliary (face-gallery identity match; per-id reference frames).

## qa/ — audits & validation (run from `full_run/pkl/`)
- `validate.py` — automated invariant checks over all pkls (schema, one-person, face/body gates, hands-near,
  provenance) + statistical flags (low-found, off-centre, jumpy). Proves correctness without watching video.
- `scan_wrist.py` / `scan_behind.py` / `scan_negev.py` / `scan_nohands*.py` — find clips exercising specific
  cases (missing wrists, behind-selection suspects, negative-evidence firing, found-but-0-hands categories).
- `classify_negev.py` — split clips into lobby (many small bodies) vs real table.
- `runner_viz.py <stems.txt> <outdir> <shard> <nshards>` — render viz for a list of clips (imports dealer_faceid).
- `render_nohands.py` — render annotated example frames per no-hands category.
- `postgate.py` — apply stricter face/body gates to existing pkls in place (drop-only; ~= a re-run, ~100x faster).

## launchers/ — shell drivers
`run_final.sh` (main run, 16 shards), `run_quality.sh` (per-clip quality filter, 20 shards),
`run_perframe.sh` (per-frame quality filter, 24 shards), `run_apply_pkl.sh` (apply quality->pkls, 16 shards),
`run_full.sh` (10-shard variant),
`run_audit_viz.sh` / `run_wrong.sh` (audit-viz sets). NOTE: they write shard logs to a scratchpad path — edit `SP`
at the top before reuse.

## Outputs
- `full_run/pkl/<stem>_dealer.pkl` (73,858) — single-dealer, drop-in schema.
- `full_run/vis/`, `vis_wrist/`, `vis_behind/`, `viis_nohand_frame/` — spot-check visualizations.
- `videos/quality_multi.parquet` — per-clip dark/blur/ocr/face usability manifest.
