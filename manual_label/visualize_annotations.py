import json
import os
import subprocess
import cv2
import glob
from typing import Dict, List, Tuple, Optional


def find_video_file(video_name: str, video_root_dir: str) -> Optional[str]:
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


def get_frame_annotation(frame_num: int, timeline_segments: List[Dict]) -> Tuple[List[Tuple[str, bool]], List[str], List[Dict]]:
    """
    Get annotation information for a specific frame.

    Args:
        frame_num: Frame number
        timeline_segments: List of timeline segments

    Returns:
        Tuple of (labels_with_bbox_flag, meta_text, drawable_bboxes)
        drawable_bboxes is a list of dicts with {x, y, width, height, labels}
        containing only the keyframes that fall on this exact frame number.
    """
    labels_with_bbox = []
    meta_text = []
    drawable_bboxes = []

    for segment in timeline_segments:
        # Check if frame is in this segment
        if segment['start_frame'] <= frame_num <= segment['end_frame']:
            has_bbox = len(segment.get('bounding_boxes', [])) > 0

            for label in segment['labels']:
                labels_with_bbox.append((label, has_bbox))

            meta_text.extend(segment['meta_text'])

            # Check each bbox's keyframes for an exact match on this frame
            for bbox in segment['bounding_boxes']:
                for kf in bbox.get('keyframes', []):
                    if kf['frame'] == frame_num:
                        drawable_bboxes.append({
                            'x': kf['x'],
                            'y': kf['y'],
                            'width': kf['width'],
                            'height': kf['height'],
                            'labels': bbox['labels'],
                        })

    return labels_with_bbox, meta_text, drawable_bboxes


def draw_text_with_background(frame, text: str, position: Tuple[int, int],
                              font_scale: float = 0.8, thickness: int = 2,
                              text_color: Tuple[int, int, int] = (255, 255, 255),
                              bg_color: Tuple[int, int, int] = (0, 0, 0),
                              padding: int = 10):
    """
    Draw text with a background rectangle.

    Args:
        frame: Video frame
        text: Text to draw
        position: (x, y) position for text
        font_scale: Font scale
        thickness: Text thickness
        text_color: RGB color for text
        bg_color: RGB color for background
        padding: Padding around text
    """
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Calculate background rectangle coordinates
    x, y = position
    rect_x1 = x - padding
    rect_y1 = y - text_height - padding
    rect_x2 = x + text_width + padding
    rect_y2 = y + baseline + padding

    # Draw background rectangle
    cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)

    # Draw text
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

    return text_height + baseline + 2 * padding


def draw_bounding_box(frame, bbox: Dict, frame_width: int, frame_height: int):
    """
    Draw a bounding box on the frame.

    Args:
        frame: Video frame
        bbox: Dict with {x, y, width, height} in percentage and optional labels
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels
    """
    # Convert percentage to pixel coordinates
    x1 = int((bbox['x'] / 100) * frame_width)
    y1 = int((bbox['y'] / 100) * frame_height)
    x2 = int(((bbox['x'] + bbox['width']) / 100) * frame_width)
    y2 = int(((bbox['y'] + bbox['height']) / 100) * frame_height)

    # Draw rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw label if available
    if bbox.get('labels'):
        label_text = ', '.join(bbox['labels'])
        label_y = max(y1 - 10, 20)
        draw_text_with_background(frame, label_text, (x1, label_y),
                                 font_scale=0.6, thickness=2,
                                 text_color=(0, 255, 0), bg_color=(0, 0, 0),
                                 padding=5)


def visualize_video(annotation_file: str, video_root_dir: str, output_dir: str = None):
    """
    Create a visualization video with temporal labels and bounding boxes.

    Args:
        annotation_file: Path to the annotation JSON file
        video_root_dir: Root directory containing videos
        output_dir: Directory to save output videos (default: annotations_dir/visualizations)
    """
    # Load annotation
    with open(annotation_file, 'r', encoding='utf-8') as f:
        annotation = json.load(f)

    video_name = annotation['video_name']
    timeline_segments = annotation['timeline_segments']

    print(f"\nProcessing: {video_name}")

    # Find video file
    video_path = find_video_file(video_name, video_root_dir)
    if video_path is None:
        print(f"  ✗ Video file not found: {video_name}")
        return

    print(f"  Found video: {video_path}")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ✗ Failed to open video: {video_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  Video info: {frame_width}x{frame_height}, {fps:.2f} fps, {total_frames} frames")

    # Set output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(annotation_file), 'visualizations')
    os.makedirs(output_dir, exist_ok=True)

    # Create output video path
    video_name_no_ext = os.path.splitext(video_name)[0]
    output_path = os.path.join(output_dir, f"{video_name_no_ext}_visualized.mp4")

    # Open ffmpeg process for H.264 encoding via pipe
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{frame_width}x{frame_height}',
        '-pix_fmt', 'bgr24',
        '-r', str(fps),
        '-i', 'pipe:0',
        '-vcodec', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '18',
        output_path,
    ]
    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Process each frame
    frame_num = 0
    processed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        # Get annotations for this frame
        labels_with_bbox, meta_text, drawable_bboxes = get_frame_annotation(frame_num, timeline_segments)

        # Draw bounding boxes (each entry is already a {x,y,width,height,labels} dict)
        for bbox in drawable_bboxes:
            draw_bounding_box(frame, bbox, frame_width, frame_height)

        # Draw temporal labels at middle bottom
        if labels_with_bbox or meta_text:
            display_texts = []
            if labels_with_bbox:
                formatted_labels = []
                for label, has_bbox in labels_with_bbox:
                    if has_bbox:
                        formatted_labels.append(f"{label} [bbox]")
                    else:
                        formatted_labels.append(label)
                display_texts.append(' | '.join(formatted_labels))
            if meta_text:
                display_texts.append(f"Meta: {' | '.join(meta_text)}")

            y_offset = frame_height - 20

            for text in display_texts:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2
                (text_width, _), _ = cv2.getTextSize(text, font, font_scale, thickness)
                x_pos = (frame_width - text_width) // 2

                height = draw_text_with_background(frame, text, (x_pos, y_offset),
                                                  font_scale=font_scale, thickness=thickness,
                                                  text_color=(255, 255, 255),
                                                  bg_color=(0, 0, 0),
                                                  padding=10)
                y_offset -= height + 5

        # Draw frame number at top-left
        frame_info = f"Frame: {frame_num}/{total_frames}"
        draw_text_with_background(frame, frame_info, (10, 30),
                                 font_scale=0.6, thickness=1,
                                 text_color=(255, 255, 0),
                                 bg_color=(0, 0, 0),
                                 padding=5)

        # Send raw frame bytes to ffmpeg
        ffmpeg_proc.stdin.write(frame.tobytes())
        processed_frames += 1

        # Progress indicator
        if processed_frames % 100 == 0:
            progress = (processed_frames / total_frames) * 100
            print(f"  Progress: {processed_frames}/{total_frames} frames ({progress:.1f}%)", end='\r')

    # Cleanup
    cap.release()
    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()

    print(f"\n  ✓ Visualization saved: {output_path}")
    print(f"  Processed {processed_frames} frames")


def process_annotations(
    video_root_dir: str,
    output_dir: str = None,
    annotations_dir: str = None,
    specific_files: List[str] = None,
    max_files: int = None,
):
    """
    Process annotation files — either a specific list of files or a whole folder.

    Args:
        video_root_dir: Root directory containing videos
        output_dir: Directory to save output videos
        annotations_dir: Directory containing *_annotations.json files (folder mode)
        specific_files: List of annotation JSON file paths to process (single/subset mode)
        max_files: If set, process at most this many files (applies to folder mode)
    """
    if specific_files:
        annotation_files = [f for f in specific_files if os.path.isfile(f)]
        missing = [f for f in specific_files if not os.path.isfile(f)]
        for m in missing:
            print(f"Warning: file not found, skipping: {m}")
    elif annotations_dir:
        annotation_files = sorted(glob.glob(os.path.join(annotations_dir, '*_annotations.json')))
        if max_files is not None:
            annotation_files = annotation_files[:max_files]
    else:
        print("Error: provide either annotations_dir or specific_files.")
        return

    if not annotation_files:
        print("No annotation files to process.")
        return

    print(f"Found {len(annotation_files)} annotation file(s) to process.")
    print("=" * 60)

    for i, annotation_file in enumerate(annotation_files, 1):
        print(f"\n[{i}/{len(annotation_files)}] {os.path.basename(annotation_file)}")
        try:
            visualize_video(annotation_file, video_root_dir, output_dir)
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("✓ All videos processed!")


if __name__ == "__main__":
    # ── Configuration ──────────────────────────────────────────────────────────
    VIDEO_ROOT_DIR   = '/home/ubuntu/yifan/code/FACT_actseg/Data_Filtering/filtered_videos'
    OUTPUT_DIR       = './anno/visualizations'

    # ── Mode: choose one ───────────────────────────────────────────────────────
    # Option A — process a whole folder (set MAX_FILES=None to process all)
    ANNOTATIONS_DIR  = './anno/good_quality_round_annotated_305_0223'
    MAX_FILES        = 10          # e.g. 5 to process only the first 5 files

    # Option B — process specific files only (overrides folder mode when non-empty)
    SPECIFIC_FILES   = [
        './anno/good_quality_round_annotated_305_0223/2025-10-05_03-48-41_010497_013126_annotations.json',
    ]
    # ──────────────────────────────────────────────────────────────────────────

    process_annotations(
        video_root_dir=VIDEO_ROOT_DIR,
        output_dir=OUTPUT_DIR,
        annotations_dir=ANNOTATIONS_DIR if not SPECIFIC_FILES else None,
        specific_files=SPECIFIC_FILES if SPECIFIC_FILES else None,
        max_files=MAX_FILES,
    )
