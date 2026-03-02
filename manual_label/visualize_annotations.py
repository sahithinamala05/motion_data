import json
import os
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
        Tuple of (labels_with_bbox_flag, meta_text, bounding_boxes)
        where labels_with_bbox_flag is a list of (label, has_bbox) tuples
    """
    labels_with_bbox = []
    meta_text = []
    bounding_boxes = []

    for segment in timeline_segments:
        # Check if frame is in this segment
        if segment['start_frame'] <= frame_num <= segment['end_frame']:
            # Check if this segment has any bounding boxes
            has_bbox = len(segment.get('bounding_boxes', [])) > 0

            # Add labels with bbox flag
            for label in segment['labels']:
                labels_with_bbox.append((label, has_bbox))

            meta_text.extend(segment['meta_text'])

            # Check for bounding boxes at this specific frame
            for bbox in segment['bounding_boxes']:
                if bbox['frame'] == frame_num:
                    bounding_boxes.append(bbox)

    return labels_with_bbox, meta_text, bounding_boxes


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
        bbox: Bounding box dictionary with x, y, width, height in percentage
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels
    """
    # Convert percentage to pixel coordinates
    x_percent = bbox['x']
    y_percent = bbox['y']
    width_percent = bbox['width']
    height_percent = bbox['height']

    # Calculate pixel coordinates
    x1 = int((x_percent / 100) * frame_width)
    y1 = int((y_percent / 100) * frame_height)
    x2 = int(((x_percent + width_percent) / 100) * frame_width)
    y2 = int(((y_percent + height_percent) / 100) * frame_height)

    # Draw rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw label if available
    if bbox.get('labels'):
        label_text = ', '.join(bbox['labels'])
        # Draw label background and text above the box
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
        output_dir: Directory to save output videos (default: same as annotation file)
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

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    if not out.isOpened():
        print(f"  ✗ Failed to create output video: {output_path}")
        cap.release()
        return

    # Process each frame
    frame_num = 0
    processed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        # Get annotations for this frame
        labels_with_bbox, meta_text, bboxes = get_frame_annotation(frame_num, timeline_segments)

        # Draw bounding boxes
        for bbox in bboxes:
            draw_bounding_box(frame, bbox, frame_width, frame_height)

        # Draw temporal labels at middle bottom
        if labels_with_bbox or meta_text:
            # Combine labels and meta text
            display_texts = []
            if labels_with_bbox:
                # Format labels with bbox indicator
                formatted_labels = []
                for label, has_bbox in labels_with_bbox:
                    if has_bbox:
                        formatted_labels.append(f"{label} [bbox]")
                    else:
                        formatted_labels.append(label)
                display_texts.append(' | '.join(formatted_labels))
            if meta_text:
                display_texts.append(f"Meta: {' | '.join(meta_text)}")

            # Calculate position (middle bottom)
            y_offset = frame_height - 20

            for text in display_texts:
                # Calculate x position to center the text
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                x_pos = (frame_width - text_width) // 2

                # Draw text with background
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

        # Write frame
        out.write(frame)
        processed_frames += 1

        # Progress indicator
        if processed_frames % 100 == 0:
            progress = (processed_frames / total_frames) * 100
            print(f"  Progress: {processed_frames}/{total_frames} frames ({progress:.1f}%)", end='\r')

    # Cleanup
    cap.release()
    out.release()

    print(f"\n  ✓ Visualization saved: {output_path}")
    print(f"  Processed {processed_frames} frames")


def process_all_annotations(annotations_dir: str, video_root_dir: str, output_dir: str = None):
    """
    Process all annotation files in a directory.

    Args:
        annotations_dir: Directory containing annotation JSON files
        video_root_dir: Root directory containing videos
        output_dir: Directory to save output videos
    """
    # Find all annotation JSON files
    annotation_files = glob.glob(os.path.join(annotations_dir, '*_annotations.json'))

    if not annotation_files:
        print(f"No annotation files found in {annotations_dir}")
        return

    print(f"Found {len(annotation_files)} annotation files")
    print("=" * 60)

    for i, annotation_file in enumerate(annotation_files, 1):
        print(f"\n[{i}/{len(annotation_files)}] Processing: {os.path.basename(annotation_file)}")
        try:
            visualize_video(annotation_file, video_root_dir, output_dir)
        except Exception as e:
            print(f"  ✗ Error processing {annotation_file}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("✓ All videos processed!")


if __name__ == "__main__":
    # Configuration
    annotations_directory = '/home/ubuntu/yifan/code/FACT_actseg/labelstudio/export_json/extracted_annotations_0111'
    video_root_directory = '/home/ubuntu/yifan/code/FACT_actseg/downloaded_videos'
    output_directory = '/home/ubuntu/yifan/code/FACT_actseg/labelstudio/export_json/extracted_annotations_0111/visualizations'

    # Process all annotations
    process_all_annotations(annotations_directory, video_root_directory, output_directory)
