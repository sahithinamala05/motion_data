import json
import os
import shutil
import glob
import cv2
from typing import Dict, List, Tuple
from collections import defaultdict


# Define the labels to keep (others will be mapped to background)
KEEP_LABELS = {
    # 'wait for bets',
    # 'close bets',
    # 'clean hand',
    # 'initial hands 1st',
    # 'initial hands 2nd',
    # 'reveal hole card',
    # 'discard'
    'call for action',
    'hit',
    'dealer hits',

}

# Labels that should be mapped to background (for reference and easy modification)
BACKGROUND_LABELS = {
    # 'call for action',
    'tap',
    'hit',
    'refer',
    'double',
    'split',
    'close bets',
    'initial hands 1st',
    # Add more labels here as needed
}


def find_video_file(video_name: str, video_root_dir: str) -> str:
    """
    Find the video file in the video root directory.

    Args:
        video_name: Name of the video file
        video_root_dir: Root directory containing videos

    Returns:
        Full path to the video file or None if not found
    """
    # Try exact match first
    video_path = os.path.join(video_root_dir, video_name)
    if os.path.exists(video_path):
        return video_path

    # Try searching recursively
    pattern = os.path.join(video_root_dir, '**', video_name)
    matches = glob.glob(pattern, recursive=True)
    if matches:
        return matches[0]

    return None


def get_video_frame_count(video_path: str) -> int:
    """
    Get the total number of frames in a video.

    Args:
        video_path: Path to video file

    Returns:
        Total frame count
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    return frame_count


def normalize_label(label: str) -> str:
    """
    Normalize label - map to background if not in KEEP_LABELS.

    Args:
        label: Original label

    Returns:
        Normalized label (either original or 'background')
    """
    if label in KEEP_LABELS:
        return label
    else:
        return 'background'


def create_frame_labels(timeline_segments: List[Dict], total_frames: int) -> List[str]:
    """
    Create per-frame label list from timeline segments.

    Args:
        timeline_segments: List of timeline segment dictionaries
        total_frames: Total number of frames in video

    Returns:
        List of labels, one per frame
    """
    # Initialize all frames as background
    frame_labels = ['background'] * (total_frames + 1)  # +1 because frames are 1-indexed

    # Process each segment
    for segment in timeline_segments:
        start_frame = segment['start_frame']
        end_frame = segment['end_frame']
        labels = segment['labels']

        # Get the first label (ignore if multiple labels per segment)
        if labels:
            label = normalize_label(labels[0])

            # Assign label to all frames in the range
            for frame_num in range(start_frame, end_frame + 1):
                if 0 <= frame_num <= total_frames:
                    # Handle overlaps: if frame already has a non-background label, keep it
                    # Otherwise assign the new label
                    if frame_labels[frame_num] == 'background':
                        frame_labels[frame_num] = label
                    # If there's an overlap with different labels, keep the first one
                    # (This is arbitrary - you can change the strategy if needed)

    return frame_labels[1:]  # Remove index 0, return 1-indexed frames


def generate_mapping_file(output_dir: str, all_labels: set) -> Dict[str, int]:
    """
    Generate mapping.txt file with label indices.

    Args:
        output_dir: Output directory
        all_labels: Set of all unique labels

    Returns:
        Dictionary mapping label to index
    """
    # Sort labels, but keep background last
    sorted_labels = sorted([l for l in all_labels if l != 'background'])
    sorted_labels.append('background')

    # Create mapping
    label_to_idx = {label: idx for idx, label in enumerate(sorted_labels)}

    # Write mapping file
    mapping_path = os.path.join(output_dir, 'mapping.txt')
    with open(mapping_path, 'w') as f:
        for idx, label in enumerate(sorted_labels):
            f.write(f"{idx} {label}\n")

    print(f"Generated mapping.txt with {len(sorted_labels)} labels")
    return label_to_idx


def generate_splits(video_names: List[str], output_dir: str, train_ratio: float = 0.8):
    """
    Generate train/test split files.

    Args:
        video_names: List of video names (without .mp4 extension)
        output_dir: Output directory for splits
        train_ratio: Ratio of training data (default 0.8 = 80% train, 20% test)
    """
    splits_dir = os.path.join(output_dir, 'splits')
    os.makedirs(splits_dir, exist_ok=True)

    # Sort video names for reproducibility
    video_names = sorted(video_names)
    total_videos = len(video_names)

    # Calculate split point
    train_count = int(total_videos * train_ratio)

    # Create single split (you can expand this to create multiple splits if needed)
    train_videos = video_names[:train_count]
    test_videos = video_names[train_count:]

    # Write train split
    train_split_path = os.path.join(splits_dir, 'train.split1.bundle')
    with open(train_split_path, 'w') as f:
        for video in train_videos:
            f.write(f"{video}.txt\n")

    # Write test split
    test_split_path = os.path.join(splits_dir, 'test.split1.bundle')
    with open(test_split_path, 'w') as f:
        for video in test_videos:
            f.write(f"{video}.txt\n")

    print(f"Generated splits: {len(train_videos)} train, {len(test_videos)} test")


def prepare_dataset(annotations_dir: str, video_root_dir: str, output_dir: str):
    """
    Prepare dataset in GTEA format.

    Args:
        annotations_dir: Directory containing annotation JSON files
        video_root_dir: Root directory containing videos
        output_dir: Output directory for the prepared dataset
    """
    print("=" * 80)
    print("PREPARING DATASET IN GTEA FORMAT")
    print("=" * 80)

    # Create output directories
    videos_dir = os.path.join(output_dir, 'videos')
    groundtruth_dir = os.path.join(output_dir, 'groundTruth')

    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(groundtruth_dir, exist_ok=True)

    # Find all annotation files
    annotation_files = glob.glob(os.path.join(annotations_dir, '*_annotations.json'))
    print(f"\nFound {len(annotation_files)} annotation files")

    all_labels = set()
    all_labels.add('background')
    processed_videos = []

    # Process each annotation file
    for i, annotation_file in enumerate(annotation_files, 1):
        print(f"\n[{i}/{len(annotation_files)}] Processing: {os.path.basename(annotation_file)}")

        # Load annotation
        with open(annotation_file, 'r') as f:
            annotation = json.load(f)

        video_name = annotation['video_name']
        timeline_segments = annotation['timeline_segments']

        # Find video file
        video_path = find_video_file(video_name, video_root_dir)
        if video_path is None:
            print(f"  ✗ Video not found: {video_name}")
            continue

        print(f"  Found video: {video_path}")

        # Get video frame count
        try:
            total_frames = get_video_frame_count(video_path)
            print(f"  Total frames: {total_frames}")
        except Exception as e:
            print(f"  ✗ Error reading video: {e}")
            continue

        # Copy video to output directory
        video_name_only = os.path.splitext(video_name)[0]
        output_video_path = os.path.join(videos_dir, video_name)

        if not os.path.exists(output_video_path):
            shutil.copy2(video_path, output_video_path)
            print(f"  Copied video to: {output_video_path}")
        else:
            print(f"  Video already exists: {output_video_path}")

        # Create frame-level labels
        frame_labels = create_frame_labels(timeline_segments, total_frames)

        # Collect unique labels
        unique_labels_in_video = set(frame_labels)
        all_labels.update(unique_labels_in_video)

        # Count labels
        label_counts = defaultdict(int)
        for label in frame_labels:
            label_counts[label] += 1

        print(f"  Label distribution:")
        for label, count in sorted(label_counts.items()):
            print(f"    {label}: {count} frames ({count/len(frame_labels)*100:.1f}%)")

        # Write groundTruth file
        groundtruth_path = os.path.join(groundtruth_dir, f"{video_name_only}.txt")
        with open(groundtruth_path, 'w') as f:
            for label in frame_labels:
                f.write(f"{label}\n")

        print(f"  ✓ Wrote groundTruth: {groundtruth_path}")

        processed_videos.append(video_name_only)

    print("\n" + "=" * 80)
    print(f"Processed {len(processed_videos)} videos successfully")

    # Generate mapping.txt
    print("\n" + "-" * 80)
    label_to_idx = generate_mapping_file(output_dir, all_labels)
    print(f"Labels: {', '.join(sorted(all_labels))}")

    # Generate splits
    print("\n" + "-" * 80)
    generate_splits(processed_videos, output_dir)

    print("\n" + "=" * 80)
    print("DATASET PREPARATION COMPLETE!")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")
    print(f"  - videos/: {len(processed_videos)} video files")
    print(f"  - groundTruth/: {len(processed_videos)} annotation files")
    print(f"  - mapping.txt: {len(all_labels)} labels")
    print(f"  - splits/: train/test splits")
    print("\nNext steps:")
    print("  1. Generate features for the videos (features/ directory)")
    print("  2. Update paths in your training configuration")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Configuration
    annotations_directory = './anno/good_quality_round_annotated_305_0223'
    video_root_directory = '/home/ubuntu/yifan/code/FACT_actseg/Data_Filtering/filtered_videos'
    output_directory = '/home/ubuntu/yifan/code/cleanpull/FACT_actseg/data/livedealer'

    # Run preparation
    prepare_dataset(annotations_directory, video_root_directory, output_directory)

    # Print label configuration info
    print("\n" + "=" * 80)
    print("LABEL CONFIGURATION")
    print("=" * 80)
    print("\nKept labels (from annotations):")
    for label in sorted(KEEP_LABELS):
        print(f"  - {label}")

    print("\nMapped to background:")
    for label in sorted(BACKGROUND_LABELS):
        print(f"  - {label}")

    print("\nTo add labels back from background:")
    print("  1. Remove the label from BACKGROUND_LABELS set")
    print("  2. Add the label to KEEP_LABELS set")
    print("  3. Re-run this script")
    print("=" * 80)
