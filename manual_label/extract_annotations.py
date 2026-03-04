import json
import os
from typing import List, Dict, Any


# Action categories for annotation validation
ACTIONS_REQUIRE_META = {"initial hand 1st", "initial hand 2nd", "discard"}
# split/align: 1 bbox annotation with exactly 2 keyframes (2 drawings at different frames)
ACTIONS_REQUIRE_2_KEYFRAMES = {"split", "align"}
ACTIONS_REQUIRE_1_BBOX = {"call for action", "hit", "dealer hits", "wave", "tap", "refer"}


def check_annotations(video_annotations: Dict[str, Any]) -> List[str]:
    """
    Check annotation rules and return a list of violation messages.

    Rules:
    1. initial hand 1st, initial hand 2nd, discard: must have non-empty meta
    2. split, align: must have exactly 1 bbox with exactly 2 keyframes per temporal segment
    3. call for action, hit, dealer hits, wave, tap, refer: must have exactly 1 bbox per segment
    """
    violations = []

    for seg in video_annotations.get('timeline_segments', []):
        labels = seg.get('labels', [])
        meta_text = seg.get('meta_text', [])
        bboxes = seg.get('bounding_boxes', [])
        start = seg.get('start_frame', '?')
        end = seg.get('end_frame', '?')

        for label in labels:
            label_key = label.lower().strip()

            # Rule 1: certain actions must have non-empty meta
            if label_key in ACTIONS_REQUIRE_META:
                if not meta_text or all(str(t).strip() == '' for t in meta_text):
                    violations.append(
                        f"Segment [{start}-{end}] label='{label}': "
                        f"meta must be non-empty"
                    )

            # Rule 2: split/align must have exactly 1 bbox with exactly 2 keyframes
            if label_key in ACTIONS_REQUIRE_2_KEYFRAMES:
                if len(bboxes) != 1:
                    violations.append(
                        f"Segment [{start}-{end}] label='{label}': "
                        f"expected 1 bbox, got {len(bboxes)}"
                    )
                else:
                    n_kf = len(bboxes[0].get('keyframes', []))
                    if n_kf != 2:
                        violations.append(
                            f"Segment [{start}-{end}] label='{label}': "
                            f"expected 2 keyframes in bbox, got {n_kf}"
                        )

            # Rule 3: certain actions must have exactly 1 bbox
            if label_key in ACTIONS_REQUIRE_1_BBOX:
                if len(bboxes) != 1:
                    violations.append(
                        f"Segment [{start}-{end}] label='{label}': "
                        f"expected 1 bbox, got {len(bboxes)}"
                    )

    return violations


def extract_video_annotations(json_path: str, output_dir: str = None) -> None:
    """
    Extract timeline labels and videorectangle labels from Label Studio JSON export.
    Creates separate JSON files for each video and a single check txt file for all videos.

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

    # Collect all violations across videos for the single check file
    all_violations: List[str] = []

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

                    # Extract all keyframes from sequence (each is a drawing at a specific frame)
                    if sequence and len(sequence) > 0:
                        anchor_frame = sequence[0].get('frame', None)

                        if anchor_frame is not None:
                            keyframes = [
                                {
                                    'frame': kf.get('frame', 0),
                                    'x': kf.get('x', 0),
                                    'y': kf.get('y', 0),
                                    'width': kf.get('width', 0),
                                    'height': kf.get('height', 0),
                                    'time': kf.get('time', 0),
                                    'rotation': kf.get('rotation', 0),
                                }
                                for kf in sequence
                            ]
                            bbox_info = {
                                'frame': anchor_frame,  # first keyframe, used for segment matching
                                'labels': labels,
                                'keyframes': keyframes,
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
                        print(f"  Matched bbox at frame {frame} ({len(bbox['keyframes'])} keyframe(s)) "
                              f"to segment {segment['labels']} ({segment['start_frame']}-{segment['end_frame']})")

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

        # Run annotation checks and collect violations
        violations = check_annotations(video_annotations)
        if violations:
            all_violations.append(f"=== {video_name} ({len(violations)} violation(s)) ===")
            all_violations.extend(violations)
            print(f"  [WARNING] {len(violations)} violation(s) found.")
        else:
            all_violations.append(f"=== {video_name}: OK ===")
            print(f"  [OK] All annotation checks passed.")

    # Write all check results to a single txt file
    input_name = os.path.splitext(os.path.basename(json_path))[0]
    check_file = os.path.join(output_dir, f"{input_name}_check.txt")
    total_violations = sum(1 for v in all_violations if not v.startswith('==='))

    with open(check_file, 'w', encoding='utf-8') as f:
        f.write(f"Annotation check for: {json_path}\n")
        f.write(f"Total videos: {len(videos_data)}  |  Total violations: {total_violations}\n")
        f.write("=" * 60 + "\n\n")
        for line in all_violations:
            f.write(line + '\n')

    print(f"\n✓ Processing complete! Processed {len(videos_data)} videos.")
    print(f"  Check report saved to: {check_file}")


if __name__ == "__main__":

    input_json = './data/good_quality_round_annotated_305_0223.json'
    output_directory = './anno/good_quality_round_annotated_305_0223'

    # Run extraction
    extract_video_annotations(input_json, output_directory)
