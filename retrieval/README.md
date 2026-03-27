# Action Detection via Sliding-Window Classification

Detect target actions (split, clean hand, etc.) in blackjack videos using pre-extracted DINOv3 frame features.

## Algorithm

1. **Representation**: Each sliding window is divided into 5 temporal bins. Per-bin mean-pooled DINOv3 features are concatenated -> 5x768 = 3840-dim descriptor capturing temporal structure.

2. **Model**: `SplitDetector` -- 2-layer MLP encoder (3840->512->256 with BN, ReLU, Dropout) + 16-class classifier head (all annotated blackjack actions). Outputs P(action) per window.

3. **Training**: Joint cross-entropy + triplet loss on sliding windows from all 634 annotated videos. Balanced sampling (25% focus action per batch). Windows labeled by >=50% overlap with annotated segments. Feature dropout (15%) for augmentation. Window size auto-computed from the action's segment length distribution (75th percentile).

4. **Inference**: Slide window over new video features -> P(action) per window -> 1D NMS -> extract side-by-side H.264 clips for detections above threshold. Training video IDs are automatically skipped.

## Code Structure

| File | Description |
|------|-------------|
| `dataset.py` | Loads annotations + DINOv3 features, builds window-level training data |
| `triplet_retrieval.py` | `SplitDetector` model, `train_detector()`, `project_items()` |
| `train.py` | Training pipeline: 5-fold stratified CV + final model, saves checkpoint |
| `inference.py` | Sliding-window detection, clip extraction, JSON output |
| `upload_predictions_new_project.py` | Upload detections as pre-labels to a new Label Studio project |
| `run_upload.sh` | Wrapper script for upload with token and project ID |

## Input / Output

### Training

**Input:**
- Annotation JSONs: `manual_label/anno/good_quality_round_annotated_634_0312/*_annotations.json`
- DINOv3 features: `filtered_videos_feat/*.npy` -- shape `(T, 768)` per video

**Output:**
- `checkpoints/detector_{action}_*.pt` -- model weights + metadata + training video IDs
- `checkpoints/detector_{action}_*.gallery.npy` -- gallery embeddings (for visualization)

### Inference

**Input:**
- Checkpoint `.pt` file (encodes action, window, stride, n_bins)
- Directory of `.npy` feature files (one per video)
- Matching directory of `.mp4` video files (for clip extraction)

**Output:**
- `vis_out/inference_{action}/detections_*.json` -- all detections with video_id, scores, frame ranges
- `vis_out/inference_{action}/*_sidebyside.mp4` -- side-by-side clips (query | nearest gallery match)

## Usage

```bash
conda activate fact

# Train split detector (auto window=25, stride=6 based on segment lengths)
python train.py --action split

# Train clean hand detector
python train.py --action "clean hand"

# Train any action (window/stride auto-computed, or override manually)
python train.py --action hit --window 40 --stride 10

# Skip cross-validation
python train.py --action split --no_cv

# Inference (window/stride auto-loaded from checkpoint)
python inference.py --ckpt checkpoints/detector_split_d256_e300_m0.3_s42.pt

# Inference on specific directory
python inference.py \
    --ckpt checkpoints/detector_clean_hand_d256_e300_m0.3_s42.pt \
    --feat_dir DATA/feat/ \
    --video_dir DATA/video/

# Adjust confidence threshold (default 0.9)
python inference.py \
    --ckpt checkpoints/detector_split_d256_e300_m0.3_s42.pt \
    --conf_threshold 0.95
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--action` | split | Target action label (any from ALL_LABELS) |
| `--epochs` | 300 | Training epochs |
| `--window` | auto | Sliding window size (auto from 75th pct of segment lengths) |
| `--stride` | auto | Sliding window stride (auto = window // 4) |
| `--n_bins` | 5 | Temporal bins per window |
| `--conf_threshold` | 0.9 | Min P(action) to save a clip |
| `--topn` | 1 | Max detections per video (after NMS) |

## Uploading Pre-labels to a New Label Studio Project

After inference, upload detections as pre-label annotations to a new Label Studio project. The script fetches tasks from the target project, re-maps video filenames to the new task IDs, and uploads timeline label annotations.

```bash
# Edit run_upload.sh with the target project ID and token, then:
bash run_upload.sh

# Or run directly:
python upload_predictions_new_project.py \
    --url https://app.humansignal.com \
    --token YOUR_REFRESH_TOKEN \
    --project-id 245163

# Dry run (preview first annotation without uploading)
python upload_predictions_new_project.py \
    --url https://app.humansignal.com \
    --token YOUR_REFRESH_TOKEN \
    --project-id 245163 \
    --dry-run
```

Detection JSONs are read from `vis_out/inference*/detections_*.json` (configured via `INPUT_FILES` in the script). Detections below `SCORE_THRESHOLD` (default 0.9) are filtered out. The `--token` is a Label Studio refresh token, automatically exchanged for an access token.
