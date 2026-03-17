# Split Action Detection via Sliding-Window Classification

Detect "split" actions in blackjack videos using pre-extracted DINOv3 frame features.

## Algorithm

1. **Representation**: Each 60-frame sliding window is divided into 5 temporal bins. Per-bin mean-pooled DINOv3 features are concatenated → 5×768 = 3840-dim descriptor capturing temporal structure (approach → place → separate).

2. **Model**: `SplitDetector` — 2-layer MLP encoder (3840→512→256 with BN, ReLU, Dropout) + 16-class classifier head (all annotated blackjack actions). Outputs P(split) per window.

3. **Training**: Joint cross-entropy + triplet loss on sliding windows from all 634 annotated videos. Balanced sampling (25% split per batch) handles class imbalance. Windows are labeled by ≥50% overlap with annotated segments. Feature dropout (15%) for augmentation.

4. **Inference**: Slide window over new video features → P(split) per window → 1D NMS → extract side-by-side H.264 clips for detections above threshold. Training video IDs are automatically skipped.

## Code Structure

| File | Description |
|------|-------------|
| `dataset.py` | Loads annotations + DINOv3 features, builds window-level training data |
| `triplet_retrieval.py` | `SplitDetector` model, `train_detector()`, `project_items()` |
| `train.py` | Training pipeline: 5-fold stratified CV + final model, saves checkpoint |
| `inference.py` | Sliding-window detection, clip extraction, JSON output |

## Input / Output

### Training

**Input:**
- Annotation JSONs: `manual_label/anno/good_quality_round_annotated_634_0312/*_annotations.json`
- DINOv3 features: `filtered_videos_feat/*.npy` — shape `(T, 768)` per video

**Output:**
- `checkpoints/detector_v3_*.pt` — model weights + metadata + training video IDs
- `checkpoints/detector_v3_*.gallery.npy` — split gallery embeddings (for visualization)

### Inference

**Input:**
- Checkpoint `.pt` file
- Directory of `.npy` feature files (one per video)
- Matching directory of `.mp4` video files (for clip extraction)

**Output:**
- `vis_out/inference/detections_*.json` — all detections with scores and frame ranges
- `vis_out/inference/*_sidebyside.mp4` — side-by-side clips (query | nearest gallery match) for detections above threshold

## Usage

```bash
# Activate environment
conda activate fact

# Train (5-fold CV + final model, ~5 min on GPU)
python train.py --epochs 300

# Inference on toy videos
python inference.py \
    --ckpt checkpoints/detector_v3_d256_e300_m0.3_s42.pt \
    --feat_dir DATA/feat/ \
    --video_dir DATA/video/

# Batch inference (skips training videos automatically)
python inference.py \
    --ckpt checkpoints/detector_v3_d256_e300_m0.3_s42.pt

# Adjust confidence threshold (default 0.9)
python inference.py \
    --ckpt checkpoints/detector_v3_d256_e300_m0.3_s42.pt \
    --conf_threshold 0.95
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 300 | Training epochs |
| `--window` | 60 | Sliding window size (frames) |
| `--stride` | 15 | Sliding window stride |
| `--n_bins` | 5 | Temporal bins per window |
| `--conf_threshold` | 0.9 | Min P(split) to save a clip |
| `--topn` | 1 | Max detections per video (after NMS) |
