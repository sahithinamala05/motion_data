import json
import os
from typing import List, Dict, Any


def extract_video_annotations(json_path: str, output_dir: str = None) -> None:
    """
    Extract timeline labels and videorectangle labels from Label Studio JSON export.
    Creates separate JSON files for each video.

    Args:
        json_path: Path to the input JSON file
        output_dir: Directory to save output files (default: same as input)
    """
    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(json_path)

    os.makedirs(output_dir, exist_ok=True)

    # Load the JSON file
    with open(json_path, 'r', encoding='utf-8') as file:
        videos_data = json.load(file)

    print(f"Processing {len(videos_data)} videos...")

    # Process each video
    for video_idx, video_data in enumerate(videos_data):
        # Extract video name
        video_name = video_data.get('file_upload', '')
        if not video_name:
            video_path = video_data.get('data', {}).get('video', '')
            video_name = os.path.basename(video_path) if video_path else f'video_{video_idx}'

        print(f"\nProcessing video: {video_name}")

        # Initialize data structure
        video_annotations = {
            'video_name': video_name,
            'video_id': video_data.get('id', ''),
            'video_path': video_data.get('data', {}).get('video', ''),
            'timeline_segments': []
        }

        # Extract annotations
        annotations = video_data.get('annotations', [])

        for annotation in annotations:
            result = annotation.get('result', [])

            # First pass: Extract all timeline labels
            timeline_segments = []
            videorectangles = []

            for item in result:
                item_type = item.get('type', '')

                if item_type == 'timelinelabels':
                    value = item.get('value', {})
                    ranges = value.get('ranges', [])
                    timeline_labels = value.get('timelinelabels', [])
                    meta = item.get('meta', {})
                    meta_text = meta.get('text', [])

                    # Process each range
                    for range_item in ranges:
                        start_frame = int(range_item['start'])
                        end_frame = int(range_item['end'])

                        segment = {
                            'start_frame': start_frame,
                            'end_frame': end_frame,
                            'labels': timeline_labels,
                            'meta_text': meta_text,
                            'duration_frames': end_frame - start_frame,
                            'id': item.get('id', ''),
                            'bounding_boxes': []  # Will be populated in second pass
                        }
                        timeline_segments.append(segment)

                elif item_type == 'videorectangle':
                    value = item.get('value', {})
                    labels = value.get('labels', [])
                    sequence = value.get('sequence', [])

                    # Extract frame and bounding box info from sequence
                    if sequence and len(sequence) > 0:
                        seq_item = sequence[0]
                        frame = seq_item.get('frame', None)

                        if frame is not None:
                            bbox_info = {
                                'frame': frame,
                                'labels': labels,
                                'x': seq_item.get('x', 0),
                                'y': seq_item.get('y', 0),
                                'width': seq_item.get('width', 0),
                                'height': seq_item.get('height', 0),
                                'time': seq_item.get('time', 0),
                                'rotation': seq_item.get('rotation', 0),
                                'id': item.get('id', '')
                            }
                            videorectangles.append(bbox_info)

            # Second pass: Match videorectangles to timeline segments
            for bbox in videorectangles:
                frame = bbox['frame']
                matched = False

                for segment in timeline_segments:
                    if segment['start_frame'] <= frame <= segment['end_frame']:
                        segment['bounding_boxes'].append(bbox)
                        matched = True
                        print(f"  Matched bbox at frame {frame} to segment {segment['labels']} ({segment['start_frame']}-{segment['end_frame']})")

                if not matched:
                    print(f"  Warning: Bounding box at frame {frame} doesn't match any timeline segment")

            # Sort segments by start frame
            timeline_segments.sort(key=lambda x: x['start_frame'])

            # Add to video annotations
            video_annotations['timeline_segments'].extend(timeline_segments)

        # Summary statistics
        total_segments = len(video_annotations['timeline_segments'])
        total_bboxes = sum(len(seg['bounding_boxes']) for seg in video_annotations['timeline_segments'])
        print(f"  Extracted {total_segments} timeline segments with {total_bboxes} bounding boxes")

        # Save to individual JSON file
        video_name_clean = os.path.splitext(video_name)[0]
        output_file = os.path.join(output_dir, f"{video_name_clean}_annotations.json")

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(video_annotations, f, indent=2, ensure_ascii=False)

        print(f"  Saved to: {output_file}")

    print(f"\n✓ Processing complete! Processed {len(videos_data)} videos.")


if __name__ == "__main__":
    # Input and output paths
    # input_json = './data/export_216695_project-216695-at-2026-01-11-20-03-89027f84.json'
    # output_directory = './round_anno/extracted_annotations_0111'

    input_json = './data/good_quality_round_annotated_305_0223.json'
    output_directory = './anno/good_quality_round_annotated_305_0223'

    # Run extraction
    extract_video_annotations(input_json, output_directory)
